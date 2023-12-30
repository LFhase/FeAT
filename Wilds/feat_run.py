import copy
import argparse
import datetime
import json
import os
import sys
import csv
from tokenize import group
import tqdm
from collections import defaultdict
from tempfile import mkdtemp

import numpy as np
import torch
import torch.optim as optim
from scheduler import initialize_scheduler

import models
from config import dataset_defaults
from feat_utils import sample_feat_domains, save_best_rfc_model, save_model, set_seed, unpack_data, sample_domains, save_best_model, \
    Logger, return_predict_fn, return_criterion, fish_step

# This is secret and shouldn't be checked into version control
os.environ["WANDB_API_KEY"]=""
# Name and notes optional
# WANDB_NAME="My first run"
# WANDB_NOTES="Smaller learning rate, more regularization."
import wandb
import traceback
torch.set_num_threads(4)

runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='FLOOD')
# General
parser.add_argument('--dataset', type=str,
                    help="Name of dataset, choose from amazon, camelyon, "
                         "civil, fmow, iwildcam")
parser.add_argument('--algorithm', type=str, default='irm',
                    help='training scheme, choose between fish or erm.')
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--data-dir', type=str,
                    help='path to data dir')
parser.add_argument('--exp-dir', type=str, default="",
                    help='path to save results of experiments')
parser.add_argument('--stratified', action='store_true', default=False,
                    help='whether to use stratified sampling for classes')

parser.add_argument('--sample_domains', type=int, default=-1)
parser.add_argument('--epochs', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=-1)
parser.add_argument('--print_iters', type=int, default=1)
parser.add_argument('--eval_iters', type=int, default=-1)
parser.add_argument('--lr', type=float, default=-1)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--group_dro_step_size','-gs', type=float, default=0.01)
parser.add_argument('--penalty_weight','-p', type=float, default=1)
parser.add_argument('--penalty_weight2','-p2', type=float, default=-1)   # if there is another penalty weight to be tuned
parser.add_argument('--eps', type=float, default=1e-4)  
# parser.add_argument('--preference_choice','-pc',type=int,default=0)
parser.add_argument('--num_workers','-nw',type=int,default=4)
parser.add_argument('--frozen', action='store_true', default=False) # whether to frozen the featurizer
parser.add_argument('-np','--need_pretrain', action='store_true')
parser.add_argument('-ifeat','--need_ifeat_pretrain', action='store_true')
parser.add_argument('-pc','--use_pretrained_clf', action='store_true')
parser.add_argument('-ri','--use_init_clf', action='store_true')
parser.add_argument('-rp','--retain_penalty', type=float, default=0.01)
parser.add_argument('-rfc','--need_rfc_pretrain', action='store_true')
parser.add_argument('-rfcls','--rfc_long_syn', action='store_true')
parser.add_argument('-rcp','--rfc_ckpt_path', type=str,default="")
parser.add_argument('-lf','--load_feat_round', type=int,default=-1)
parser.add_argument('-pr','--pretrain_rounds',type=int,default=2)
parser.add_argument('-pi','--pretrain_iters',type=int,default=-1)
parser.add_argument('--use_old', action='store_true')
parser.add_argument('--no_plot', action='store_true')
parser.add_argument('--no_test', action='store_true')
parser.add_argument('--opt', type=str, default='')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--no_sch', action='store_true')    # not to use any scheduler
parser.add_argument('--scheduler', type=str, default='')
parser.add_argument('--adjust_lr', '-alr',action='store_true', default=False) # whether to adjust lr as scheduled after pretraining
# Computation
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed, set as -1 for random.')
parser.add_argument('--no_wandb', action='store_true', default=False) # whether not to use wandb
parser.add_argument('--no_drop_last', action='store_true', default=True) # whether not to drop last batch


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
# overwrite some arguments
batch_size = args.batch_size
print_iters = args.print_iters
pretrain_iters = args.pretrain_iters
epochs = args.epochs
optimiser = args.opt
args_dict = args.__dict__
args_dict.update(dataset_defaults[args.dataset])
args = argparse.Namespace(**args_dict)

if len(args.exp_dir) == 0:
    args.exp_dir = args.data_dir
os.environ["WANDB_DIR"] = args.exp_dir

# experiment directory setup
args.experiment = f"{args.dataset}_{args.algorithm}" \
    if args.experiment == '.' else args.experiment
directory_name = os.path.join(args.exp_dir,'experiments/{}'.format(args.experiment))
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
runPath = mkdtemp(prefix=runId, dir=directory_name)


new_pretrain_iters = False
if batch_size>0:
    args.batch_size=batch_size
if print_iters>0:
    args.print_iters = print_iters
if pretrain_iters>0:
    new_pretrain_iters =  args.pretrain_iters != pretrain_iters
    args.pretrain_iters = pretrain_iters
if len(optimiser)>0:
    args.optimiser = optimiser
    


exp_name = f"{args.experiment}"
if len(args.exp_name)>0:
    args.exp_name ="_"+args.exp_name
if args.use_pretrained_clf:
    args.model_name = '-pc'
elif args.use_init_clf:
    args.model_name = '-ri'
else:
    pass
if args.need_ifeat_pretrain:
    exp_name += f"_ifeat{args.model_name}_rp{args.retain_penalty}_pi{args.pretrain_iters}"
elif args.need_rfc_pretrain:
    exp_name += f"_rfc{args.model_name}_r{args.pretrain_rounds}_pi{args.pretrain_iters}"
    if args.rfc_long_syn:
        exp_name += "_ls"
else:
    exp_name += f"_erm"+(f"_pi{args.pretrain_iters}" if new_pretrain_iters else "")

if args.penalty_weight2 > 0:
    exp_name += f"_p{args.penalty_weight}p2{args.penalty_weight2}"
else:
    if args.algorithm.lower() == "groupdro":
        exp_name += f"_p{args.group_dro_step_size}"
    else:
        exp_name += f"_p{args.penalty_weight}"
    
if args.sample_domains>0:
    exp_name += f"_meta{args.sample_domains}"
    args.meta_steps = args.sample_domains
if args.frozen:
    exp_name += "_frozen"
if epochs>0:
    exp_name += f"_ep{epochs}"
    args.epochs = epochs
exp_name += f"_{args.optimiser}_lr{args.lr}{args.exp_name}_seed{args.seed}"

os.environ["WANDB_NAME"]=exp_name.replace("_","/")
group_name = "/".join(exp_name.split("_")[:-1]) # seed don't participate in the grouping

dataset_name_wandb = args.dataset
if not args.no_wandb:
    wandb_run = wandb.init(project=dataset_name_wandb, entity="cu_husky",group=group_name,id=wandb.util.generate_id())
    wandb.config = args


# Choosing and saving a random seed for reproducibility
if args.seed == -1:
    args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
set_seed(args.seed)

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

# load model
modelC = getattr(models, args.dataset)
train_loader, tv_loaders,dataset = modelC.getDataLoaders(args, device=device)
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']
model = modelC(args, weights=None).to(device)

# assert args.optimiser in ['SGD', 'Adam'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
if args.optimiser.lower() in ['sgd','adam']:
    opt = getattr(optim, args.optimiser)
elif args.optimiser.lower() == 'adai':
    from adai import Adai
    opt = Adai
else:
    raise Exception("Invalid choice of optimiser")
if args.lr>0:
    args.optimiser_args['lr'] = args.lr
# pop up unnecessary configs
if args.optimiser.lower() not in ['adam']:
    if 'amsgrad' in args.optimiser_args.keys():
        args.optimiser_args.pop('amsgrad')
    if 'betas' in args.optimiser_args.keys():
        args.optimiser_args.pop('betas')
if args.momentum > 0:
    args.optimiser_args['momentum'] = args.momentum
overall_step = 0
classifier = model.classifier
trainable_params = classifier.parameters() if args.frozen else model.parameters()
optimizer = opt(trainable_params, **args.optimiser_args)
predict_fn, criterion = return_predict_fn(args.dataset), return_criterion(args.dataset)


if args.algorithm not in ['erm'] and not args.adjust_lr:
    n_train_steps = train_loader.dataset.training_steps*args.epochs
else:
    n_train_steps = len(train_loader) * args.epochs 
if (args.need_pretrain  or args.need_rfc_pretrain or args.need_ifeat_pretrain)\
     and args.pretrain_iters>0 and not args.use_old:
    print(f"overall training steps: {n_train_steps}")
    n_train_steps += args.pretrain_iters

if args.no_sch:
    args.scheduler = None

if args.scheduler is not None and len(args.scheduler)>0:
    scheduler = initialize_scheduler(args, optimizer, n_train_steps)
else:
    scheduler = None

if args.adjust_lr:
    print("Adjusting learning rate as scheduled after pretraining...")
    n_iters = 0
    pretrain_iters = args.pretrain_iters
    pretrain_epochs = int(np.ceil(pretrain_iters/len(train_loader)))
    pbar = tqdm.tqdm(total = pretrain_iters)
    for epoch in range(pretrain_epochs):
        for i in range(len(train_loader)):
            if scheduler is not None and scheduler.step_every_batch:
                scheduler.step()
            # display progress
            pbar.set_description(f"Pretrain {n_iters}/{pretrain_iters} iters")
            pbar.update(1)
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
elif (args.need_pretrain or args.need_rfc_pretrain or args.need_ifeat_pretrain) \
    and args.pretrain_iters>0 and args.use_old:
    if args.scheduler is not None and len(args.scheduler)>0:
        try:
            if 'num_warmup_steps' in  args.scheduler_kwargs.keys():
                args.scheduler_kwargs['num_warmup_steps'] = 0
        except Exception as e:
            print(e)
        scheduler = initialize_scheduler(args, optimizer, n_train_steps)
    else:
        scheduler = None
print(optimizer,scheduler)


def train_erm(train_loader, epoch, agg):
    global overall_step
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        overall_step += 1
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and args.algorithm != 'fish':
            if not args.no_wandb:
                wandb.log({ "loss": loss.item()},step=overall_step)
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['losses'].append([running_loss / args.print_iters])
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]},step=overall_step)
                    wandb.log({"test_acc":agg['test_stat'][-1]},step=overall_step)
                running_loss=0
                model.train()
                save_best_model(model, runPath, agg, args)



from wilds.common.utils import split_into_groups
import torch.autograd as autograd
import torch.nn.functional as F
scale = torch.tensor(1.).to(device).requires_grad_()
def irm_penalty(losses, pos=-1, adjust=False):
    grad_1 = autograd.grad(losses[0::2].mean(), [scale], create_graph=True)[0]
    grad_2 = autograd.grad(losses[1::2].mean(), [scale], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)
    if pos>0 and not adjust:
        result += pos
    if result<0 and adjust:
        grad = autograd.grad(losses.mean(), [scale], create_graph=True)[0]
        result = torch.sum(grad.pow(2))
    return result

def train_irm(train_loader, epoch, agg):
    global overall_step
    model.train()
    train_loader.dataset.reset_batch()
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    running_loss = 0
    total_iters = len(train_loader)
    running_losses = []
    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        model.train()
        i += 1
        overall_step += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()
        avg_loss = 0.
        penalty = 0.
        losses_bygroup = []

        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            y_hat = model(x,frozen_mode=args.frozen)
            loss = F.cross_entropy(scale * y_hat,y,reduction="none")
            losses_bygroup.append(loss.mean())
            penalty += irm_penalty(loss,adjust=True)
            avg_loss += loss.mean()
        avg_loss /= args.meta_steps
        penalty /= args.meta_steps
        losses = [avg_loss, penalty, torch.stack(losses_bygroup).var()]
        if len(running_losses)==0:
            running_losses = [0]*len(losses)
        for (j,ll) in enumerate(running_losses):
            running_losses[j]+=losses[j].item()
        optimizer.zero_grad()
        loss = avg_loss+args.penalty_weight*penalty
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        running_loss += loss.item()

        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1

        if i % args.print_iters == 0 and args.print_iters != -1:            
            print(avg_loss,penalty)
            agg['losses'].append([l / args.print_iters for l in running_losses])
            if not args.no_wandb:
                wandb.log({ "loss": loss.item(),
                            "erm_loss": agg['losses'][-1][0],
                            "irm_loss": agg['losses'][-1][1],
                            "vrex_loss": agg['losses'][-1][2],
                            },step=overall_step)
            running_losses = [0]*len(losses)
            # agg['losses'].append([l.item() for l in losses])
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]},step=overall_step)
                    wandb.log({"test_acc":agg['test_stat'][-1]},step=overall_step)
                model.train()
                save_best_model(model, runPath, agg, args)
def train_irmx(train_loader, epoch, agg):
    global overall_step
    model.train()
    train_loader.dataset.reset_batch()
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    running_loss = 0
    total_iters = len(train_loader)
    running_losses = []
    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        model.train()
        overall_step += 1
        i += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()
        avg_loss = 0.
        penalty = 0.
        losses_bygroup = []

        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            y_hat = model(x,frozen_mode=args.frozen)
            loss = F.cross_entropy(scale * y_hat,y,reduction="none")
            losses_bygroup.append(loss.mean())
            penalty += irm_penalty(loss)
            avg_loss += loss.mean()
        avg_loss /= args.meta_steps
        penalty /= args.meta_steps
        losses = [avg_loss, penalty, torch.stack(losses_bygroup).var()]
        if len(running_losses)==0:
            running_losses = [0]*len(losses)
        for (j,ll) in enumerate(running_losses):
            running_losses[j]+=losses[j].item()
        optimizer.zero_grad()
        if args.penalty_weight2 > 0:
            loss = avg_loss+args.penalty_weight*penalty+args.penalty_weight2*torch.stack(losses_bygroup).var()
        else:
            loss = avg_loss+args.penalty_weight*(penalty+torch.stack(losses_bygroup).var())

        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        running_loss += loss.item()

        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1

        if i % args.print_iters == 0 and args.print_iters != -1:            
            print(avg_loss,penalty)
            agg['losses'].append([l / args.print_iters for l in running_losses])
            if not args.no_wandb:
                wandb.log({ "loss": loss.item(),
                            "erm_loss": agg['losses'][-1][0],
                            "irm_loss": agg['losses'][-1][1],
                            "vrex_loss": agg['losses'][-1][2],
                            },step=overall_step)
            running_losses = [0]*len(losses)
            # agg['losses'].append([l.item() for l in losses])
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]},step=overall_step)
                    wandb.log({"test_acc":agg['test_stat'][-1]},step=overall_step)
                model.train()
                save_best_model(model, runPath, agg, args)

def train_vrex(train_loader, epoch, agg):
    global overall_step
    model.train()
    train_loader.dataset.reset_batch()
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    running_loss = 0
    total_iters = len(train_loader)
    running_losses = []
    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        model.train()
        overall_step += 1
        i += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()
        # print(domains)
        avg_loss = 0.
        penalty = 0.
        losses_bygroup = []

        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            y_hat = model(x,frozen_mode=args.frozen)
            loss = F.cross_entropy(scale * y_hat,y,reduction="none")
            losses_bygroup.append(loss.mean())

            penalty += irm_penalty(loss)
            avg_loss += loss.mean()
        avg_loss /= args.meta_steps
        penalty /= args.meta_steps
        losses = [avg_loss, penalty, torch.stack(losses_bygroup).var()]
        if len(running_losses)==0:
            running_losses = [0]*len(losses)
        for (j,ll) in enumerate(running_losses):
            running_losses[j]+=losses[j].item()

        optimizer.zero_grad()
        loss = avg_loss+args.penalty_weight*torch.stack(losses_bygroup).var()
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        running_loss += loss.item()

        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1

        if i % args.print_iters == 0 and args.print_iters != -1:
            print(avg_loss,penalty)
            agg['losses'].append([l / args.print_iters for l in running_losses])
            if not args.no_wandb:
                wandb.log({ "loss": loss.item(),
                "erm_loss": agg['losses'][-1][0],
                "irm_loss": agg['losses'][-1][1],
                "vrex_loss": agg['losses'][-1][2],
                },step=overall_step)
            running_losses = [0]*len(losses)

            agg['train_loss'].append(running_loss / args.print_iters)
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]},step=overall_step)
                    wandb.log({"test_acc":agg['test_stat'][-1]},step=overall_step)
                model.train()
                save_best_model(model, runPath, agg, args)


def train_groupdro(train_loader, epoch, agg):
    global overall_step
    
    model.train()
    train_loader.dataset.reset_batch()
    if model.group_weights is None:
        group_weights = torch.zeros(len(train_loader.dataset.domain_indices)+1).to(device)
        group_weights.requires_grad_(False)
        model.group_weights = nn.Parameter(group_weights)
        model.group_weights.requires_grad_(False)
        model.group_weights[:] = 1/train_loader.dataset.num_envs
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    running_loss = 0
    total_iters = len(train_loader)
    running_losses = []
    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        model.train()
        overall_step += 1
        i += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()
        avg_loss = 0.
        penalty = 0.
        losses_bygroup = []

        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            y_hat = model(x,frozen_mode=args.frozen)
            loss = F.cross_entropy(scale * y_hat,y,reduction="none")
            losses_bygroup.append(loss.mean())

            penalty += irm_penalty(loss)
            avg_loss += loss.mean()
        avg_loss /= args.meta_steps
        penalty /= args.meta_steps
        losses = [avg_loss, penalty, torch.stack(losses_bygroup).var()]
        if len(running_losses)==0:
            running_losses = [0]*len(losses)
        for (j,ll) in enumerate(running_losses):
            running_losses[j]+=losses[j].item()
        losses_bygroup = torch.stack(losses_bygroup)
        model.group_weights[domains] = model.group_weights[domains] * torch.exp(args.group_dro_step_size*losses_bygroup.data)
        model.group_weights[domains] = (model.group_weights[domains]/(model.group_weights.sum()))
        optimizer.zero_grad()
        loss =  losses_bygroup @ model.group_weights[domains]
        loss.backward()
        optimizer.step()
        
        if scheduler is not None and scheduler.step_every_batch:
            scheduler.step()
        running_loss += loss.item()

        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1

        if i % args.print_iters == 0 and args.print_iters != -1:
            print(avg_loss,loss)
            agg['losses'].append([l / args.print_iters for l in running_losses])
            if not args.no_wandb:
                wandb.log({ "loss": loss.item(),
                "erm_loss": agg['losses'][-1][0],
                "irm_loss": agg['losses'][-1][1],
                "vrex_loss": agg['losses'][-1][2],
                },step=overall_step)
            running_losses = [0]*len(losses)
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            if i % args.eval_iters == 0 and args.eval_iters != -1:    
                test(val_loader, agg, loader_type='val')
                test(test_loader, agg, loader_type='test')
                if not args.no_wandb:
                    wandb.log({"val_acc":agg['val_stat'][-1]},step=overall_step)
                    wandb.log({"test_acc":agg['test_stat'][-1]},step=overall_step)
                model.train()
                save_best_model(model, runPath, agg, args)


def pretrain(train_loader, pretrain_iters, save_path=None):
    aggP = defaultdict(list)
    aggP['val_stat'] = [0.]
    global overall_step
    global optimizer
    global scheduler
    global model
    trainable_params = model.parameters()
    optimizer = opt([{'params':list(trainable_params),'initial_lr':args.optimiser_args['lr']}], **args.optimiser_args)
    if args.scheduler is not None and len(args.scheduler)>0:
        scheduler = initialize_scheduler(args, optimizer, n_train_steps,last_epoch=overall_step-1)
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return f"{param_group['lr'],param_group['initial_lr']}"
        print(get_lr(optimizer),scheduler.last_epoch)
    else:
        scheduler = None
    n_iters = 0
    pretrain_epochs = int(np.ceil(pretrain_iters/len(train_loader)))
    pbar = tqdm.tqdm(total = pretrain_iters)
    for epoch in range(pretrain_epochs):
        for i, data in enumerate(train_loader):
            if i==0:
                print(data[-1])
            overall_step += 1
            model.train()
            x, y = unpack_data(data, device)
            y_hat, feat = model(x,get_feat=True)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler.step_every_batch:
                scheduler.step()
            
            # display progress
            progress_str = f"Epoch {epoch} Pretrain {n_iters}/{pretrain_epochs}"+ \
                                    f"iters: Aug loss {loss.item():0.5e}"
            pbar.set_description(f"Pretrain {n_iters}/{pretrain_iters} iters: loss {loss.item():0.5e}")
            pbar.update(1)
            if not args.no_wandb:
                wandb.log({ "aug_loss": loss.item(),
                            "retain_loss": 0,
                            },step=overall_step)
            if n_iters%1 == 0:
                if scheduler is not None:
                    def get_lr(optimizer):
                        for param_group in optimizer.param_groups:
                            return f"{param_group['lr'],param_group['initial_lr']}"
                    progress_str += f" lr: {get_lr(optimizer)} last epoch: {scheduler.last_epoch}"
                print(progress_str)
            n_iters += 1
            if overall_step % args.eval_iters == 0 and args.eval_iters != -1:
                test(val_loader, aggP, loader_type='val', verbose=False)
                test(test_loader, aggP, loader_type='test', verbose=False)
                if not args.no_wandb:
                    wandb.log({"feat_val_acc":aggP['val_stat'][-1]},step=overall_step)
                    wandb.log({"feat_test_acc":aggP['test_stat'][-1]},step=overall_step)
                if save_path is None:
                    save_path = runPath
                save_best_model(model, save_path, aggP, args,pretrain=new_pretrain_iters)

            if n_iters == pretrain_iters:
                print("Pretrain is done!")
                test(val_loader, aggP, loader_type='val', verbose=False)
                test(test_loader, aggP, loader_type='test', verbose=False)
                if not args.no_wandb:
                    wandb.log({"feat_val_acc":aggP['val_stat'][-1]},step=overall_step)
                    wandb.log({"feat_test_acc":aggP['test_stat'][-1]},step=overall_step)
                if save_path is None:
                    save_path = runPath
                save_best_model(model, save_path, aggP, args,pretrain=new_pretrain_iters)
                break
        save_model(model, f'{save_path}/model_erm_ep{epoch}.rar')
        if scheduler is not None and not scheduler.step_every_batch:
            scheduler.step()
    save_model(model, f'{save_path}/model_erm_ep:last.rar')
    pbar.close()
    
    overall_step = 0
    set_seed(args.seed)
    modelC = getattr(models, args.dataset)
    train_loader, tv_loaders, dataset = modelC.getDataLoaders(args, device=device)
    model = modelC(args, weights=None).to(device)
    model.load_state_dict(torch.load(save_path + '/model'+(f"_pi{args.pretrain_iters}" if new_pretrain_iters else "")+'.rar'))
    print('Finished ERM pre-training!')
    classifier = model.classifier
    trainable_params = classifier.parameters() if args.frozen else model.parameters()
    optimizer = opt([{'params':list(trainable_params),'initial_lr':args.optimiser_args['lr']}], **args.optimiser_args)
    if args.scheduler is not None and len(args.scheduler)>0:
        scheduler = initialize_scheduler(args, optimizer, n_train_steps,last_epoch=overall_step-1)
    else:
        scheduler = None

    

import torch.nn as nn
from torch.utils.data import DataLoader

def agg_feat_weights(cur_model,classifiers,hidden_dim,num_classes,round_i):
    with torch.no_grad():
        w = cur_model.classifier.weight.data.clone().detach()
        b = cur_model.classifier.bias.data.clone().detach()
        for clf in classifiers:
            w += clf.weight.data
            b += clf.bias.data
        w /= len(classifiers)+1
        b /= len(classifiers)+1
        cur_clf_weight = cur_model.classifier.weight.clone().detach()
        cur_clf_bias = cur_model.classifier.bias.clone().detach()
        cur_model.classifier.weight.data = w
        cur_model.classifier.bias.data = b
    return cur_model, (cur_clf_weight,cur_clf_bias)

def ifeat_train(train_loader, pretrain_iters, pretrain_rounds=2, save_path=None):
    global model, overall_step, tv_loaders, optimizer, scheduler
    if save_path is None:
        save_path = runPath
    aggP = defaultdict(list)
    aggP['val_stat'] = [0.]
    
    if args.load_feat_round>0:
        print(f"Load previously trained model in round {args.load_feat_round}")
        feat_path = f'{save_path}/model_ifeat_round_{args.load_feat_round}.rar'
        model.load_state_dict(torch.load(feat_path))
    # get the loaders
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False} \
        if device.type == "cuda" else {}
    feat_train_loader = DataLoader(train_loader.dataset, \
                                batch_size=args.batch_size*20, shuffle=False, **kwargs)
        

    n_iters = 0
    pretrain_iters_per_round = int(np.ceil(pretrain_iters/pretrain_rounds))
    classifiers = []
    final_classifier = None
    train_loader_iter = iter(train_loader)
    train_loader.dataset.reset_feat_batch(train_loader_iter=train_loader_iter)

    pbar = tqdm.tqdm(total = pretrain_iters)
    last_feat_step = 0
    init_clf = (model.classifier.weight.data.clone().detach(), model.classifier.bias.data.clone().detach())
    for round_i in range(pretrain_rounds):
        hidden_dim = model.classifier.weight.size(1)
        num_classes = model.classifier.weight.size(0)
        
        if args.use_pretrained_clf:
            if round_i == 0:
                model.classifier.weight.data = init_clf[0].clone().detach()
                model.classifier.bias.data = init_clf[1].clone().detach()
            else:
                model.classifier.weight.data = classifiers[-1].weight.data.clone().detach()
                model.classifier.bias.data = classifiers[-1].bias.data.clone().detach()
        elif args.use_init_clf:
            model.classifier.weight.data = init_clf[0].clone().detach()
            model.classifier.bias.data = init_clf[1].clone().detach()
        else:
            model.classifier = nn.Linear(hidden_dim,num_classes).to(device)
        best_clf = None
        last_feat_step = len(aggP['val_stat'])-1

        optimizer = opt([{'params':list(model.parameters()),'initial_lr':args.optimiser_args['lr']}], \
                            **args.optimiser_args)
        if args.scheduler is not None and len(args.scheduler)>0:
            scheduler = initialize_scheduler(args, optimizer, n_train_steps)
        else:
            scheduler = None
        
        num_feat_domains = len(train_loader.dataset.feat_domains)
        pbar.reset()
        for pre_iter in range(pretrain_iters_per_round):
            if sum([l >= 1 for l in train_loader.dataset.feat_batches_left.values()]) < num_feat_domains:
                if num_feat_domains==1:
                    train_loader_iter = iter(train_loader)
                    train_loader.dataset.reset_feat_batch(train_loader_iter=train_loader_iter)
                else:
                    train_loader.dataset.reset_feat_batch()
                if scheduler is not None and not scheduler.step_every_batch:
                    scheduler.step()
            
            model.train()
            overall_step += 1
            # sample `meta_steps` number of domains to use for the inner loop
            domains = list(range(num_feat_domains))
            losses_aug = []
            losses_retain = []
            avg_loss = 0
            
            for (di,domain) in enumerate(domains):
                data = train_loader.dataset.get_feat_batch(domain)
                x, y = unpack_data(data, device)
                y_hat, feat = model(x,get_feat=True)
                loss_aug = F.cross_entropy(y_hat,y,reduction="none")
                losses_aug.append(loss_aug.mean())
                if domain%2==1:
                    y_hat_retain = classifiers[domain//2](feat)
                    loss_retain = F.cross_entropy(y_hat_retain,y,reduction="none")
                    losses_retain.append(loss_retain.mean())
                avg_loss += loss_aug.mean()

            loss = max(losses_aug).view(1)
            if len(losses_retain)>0:
                retain_loss = torch.stack(losses_retain).mean()*args.retain_penalty
            else:
                retain_loss = torch.zeros(1).to(device)
            loss += retain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler.step_every_batch:
                scheduler.step()

            # log the number of batches left for each domain
            for domain in domains:
                train_loader.dataset.feat_batches_left[domain]  -= 1
            n_iters += 1
            # display progress
            progress_str = f"Round {round_i} Pretrain {pre_iter}/{pretrain_iters_per_round}"+ \
                                    f"iters: Aug loss {max(losses_aug).item():0.5e} Retain loss {retain_loss.item():0.5e}"
            if not args.no_wandb:
                wandb.log({ "aug_loss": max(losses_aug).item(),
                            "retain_loss": retain_loss.item()/(args.retain_penalty+1e-6),
                            },step=overall_step)
            pbar.set_description(progress_str)
            pbar.update(1)
            if pre_iter%100 == 0:
                if scheduler is not None:
                    def get_lr(optimizer):
                        for param_group in optimizer.param_groups:
                            return param_group['lr']
                    progress_str += f" lr: {get_lr(optimizer)}"
                print(progress_str)

            if (pre_iter + 1) % args.eval_iters == 0 and args.eval_iters != -1 and pre_iter+1<pretrain_iters_per_round:
                model, cur_clf = agg_feat_weights(model,classifiers,hidden_dim,num_classes,round_i)
                test(val_loader, aggP, loader_type='val', verbose=False)
                test(test_loader, aggP, loader_type='test', verbose=False)
                if not args.no_wandb:
                    wandb.log({"feat_val_acc":aggP['val_stat'][-1]},step=overall_step)
                    wandb.log({"feat_test_acc":aggP['test_stat'][-1]},step=overall_step)
                save_best_model(model, save_path, aggP, args, ifeat=True)
                if aggP['val_stat'][-1] > max(aggP['val_stat'][:-1]):
                    best_clf = cur_clf
                model.classifier.weight.data = cur_clf[0]
                model.classifier.bias.data = cur_clf[1]
                model.train()
            if n_iters == pretrain_iters:
                print("Pretrain is done!")
                break
        model, cur_clf = agg_feat_weights(model,classifiers,hidden_dim,num_classes,round_i)
        test(val_loader, aggP, loader_type='val', verbose=False)
        test(test_loader, aggP, loader_type='test', verbose=False)
        if not args.no_wandb:
            wandb.log({"feat_val_acc":aggP['val_stat'][-1]},step=overall_step)
            wandb.log({"feat_test_acc":aggP['test_stat'][-1]},step=overall_step)
        save_best_model(model, save_path, aggP, args, ifeat=True)
        if aggP['val_stat'][-1] > max(aggP['val_stat'][:-1]):
            best_clf = cur_clf
        # last epoch model saving
        save_model(model, f"{save_path}/model_ifeat{args.model_name}_round_{round_i}_rp{args.retain_penalty}_pi{args.pretrain_iters}.rar")
        
        if best_clf is None:
            print(f"No better model found in round {round_i}, Using the last trained model")
        else:
            model.load_state_dict(torch.load(save_path + f"/model_ifeat{args.model_name}_rp{args.retain_penalty}_pi{args.pretrain_iters}.rar"))
        best_agg = torch.load(save_path + f"/losses_ifeat{args.model_name}_rp{args.retain_penalty}_pi{args.pretrain_iters}.rar")
        print(f"Round {round_i} Ends: Val acc {best_agg['val_stat'][-1]} Test Acc {best_agg['test_stat'][-1]}")
        # best epoch model saving
        save_model(model, f"{save_path}/model_ifeat{args.model_name}_round_{round_i}_rp{args.retain_penalty}_pi{args.pretrain_iters}_best.rar")
        if round_i+1<args.pretrain_rounds:
            # eval and augment new groups
            model.eval()
            correct_pred = []
            with torch.no_grad():
                for data in tqdm.tqdm(feat_train_loader):
                    x, y = unpack_data(data, device)
                    y_hat = model(x).argmax(-1)
                    correct_pred += (y_hat==y).cpu().tolist()
            correct_pred = torch.tensor(correct_pred).bool()
            new_feat_domains = [torch.nonzero(correct_pred,as_tuple=True)[0],\
                                torch.nonzero(~correct_pred,as_tuple=True)[0]]
            train_loader.dataset.replace_feat_domains(new_feat_domains)
            train_loader.dataset.reset_feat_batch()
            model.classifier.eval()
            classifiers = [model.classifier]
    train_loader.dataset.clean_feat_domains()
    set_seed(args.seed)
    modelC = getattr(models, args.dataset)
    train_loader, tv_loaders, dataset = modelC.getDataLoaders(args, device=device)
    model = modelC(args, weights=None).to(device)
    model.load_state_dict(torch.load(save_path + f"/model_ifeat{args.model_name}_rp{args.retain_penalty}_pi{args.pretrain_iters}.rar"))
    print('Finished ifeat pre-training!')
    classifier = model.classifier
    trainable_params = classifier.parameters() if args.frozen else model.parameters()
    optimizer = opt([{'params':list(trainable_params),'initial_lr':args.optimiser_args['lr']}], **args.optimiser_args)
    if args.scheduler is not None and len(args.scheduler)>0:
        scheduler = initialize_scheduler(args, optimizer, n_train_steps,last_epoch=overall_step-1)
    else:
        scheduler = None
    overall_step = 0
    pbar.close()
    

    

def agg_rfc_weights(cur_model,classifiers,hidden_dim,num_classes):
    with torch.no_grad():
        w = cur_model.classifier.weight.data.clone().detach()
        b = cur_model.classifier.bias.data.clone().detach()
        for clf in classifiers:
            w += clf.weight.data
            b += clf.bias.data
        w /= len(classifiers)
        b /= len(classifiers)
        cur_model.classifier.weight.data = w
        cur_model.classifier.bias.data = b
    return cur_model

def rfc_train(train_loader, pretrain_iters, pretrain_rounds=2, save_path=None):
    global model, overall_step, tv_loaders, optimizer, scheduler
    if save_path is None:
        save_path = runPath
    aggP = defaultdict(list)
    aggP['val_stat'] = [0.]
    
    if args.load_feat_round>0:
        print(f"Load previously trained model in round {args.load_feat_round}")
        feat_path = f"{save_path}/model_feat{args.model_name}_round_{args.load_feat_round}.rar"
        model.load_state_dict(torch.load(feat_path))
    # get the loaders
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False} \
        if device.type == "cuda" else {}
    feat_train_loader = DataLoader(train_loader.dataset, \
                                batch_size=args.batch_size*20, shuffle=False, **kwargs)
    n_iters = 0
    pretrain_iters_per_round = int(np.ceil(pretrain_iters/pretrain_rounds))
    classifiers = []
    final_classifier = None
    last_round_step = 0

    train_loader_iter = iter(train_loader)
    train_loader.dataset.reset_feat_batch(train_loader_iter=train_loader_iter)

    pbar = tqdm.tqdm(total = pretrain_iters)
    for round_i in range(pretrain_rounds):
        # re-initialize a model at each RFC round
        if round_i>0:
            model = modelC(args, weights=None).to(device)
        best_clf = None
        last_round_step = len(aggP['val_stat'])
        optimizer = opt([{'params':list(model.parameters()),'initial_lr':args.optimiser_args['lr']}], \
                            **args.optimiser_args)
        
        if args.scheduler is not None and len(args.scheduler)>0:
            scheduler = initialize_scheduler(args, optimizer, n_train_steps)
        else:
            scheduler = None
        
        pbar.reset()
        num_feat_domains = len(train_loader.dataset.feat_domains)
        for pre_iter in range(pretrain_iters_per_round):
            if sum([l >= 1 for l in train_loader.dataset.feat_batches_left.values()]) < num_feat_domains:
                if num_feat_domains==1:
                    train_loader_iter = iter(train_loader)
                    train_loader.dataset.reset_feat_batch(train_loader_iter=train_loader_iter)
                else:
                    train_loader.dataset.reset_feat_batch()
                print(train_loader.dataset.feat_batch_indices[0][0])
                
                if scheduler is not None and not scheduler.step_every_batch:
                    scheduler.step()
            model.train()
            overall_step += 1

            domains = list(range(num_feat_domains)) 
            losses_aug = []
            losses_retain = []
            avg_loss = 0
            for (di,domain) in enumerate(domains):
                data = train_loader.dataset.get_feat_batch(domain)
                x, y = unpack_data(data, device)
                y_hat, feat = model(x,get_feat=True)
                loss_aug = F.cross_entropy(y_hat,y,reduction="none")
                losses_aug.append(loss_aug.mean())
                avg_loss += loss_aug.mean()

            loss = max(losses_aug).view(1)  # only have the aug loss in RFC
            if len(losses_retain)>0:
                retain_loss = torch.stack(losses_retain).mean()*args.retain_penalty
            else:
                retain_loss = torch.zeros(1).to(device)
            loss += retain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler.step_every_batch:
                scheduler.step()

            # log the number of batches left for each domain
            for domain in domains:
                train_loader.dataset.feat_batches_left[domain] -= 1
            n_iters += 1
            # display progress
            progress_str = f"Round {round_i} Pretrain {pre_iter}/{pretrain_iters_per_round}"+ \
                                    f"iters: Aug loss {max(losses_aug).item():0.5e} Retain loss {retain_loss.item():0.5e}"
            if not args.no_wandb:
                wandb.log({ "aug_loss": max(losses_aug).item(),
                            "retain_loss": retain_loss.item()/(args.retain_penalty+1e-6),
                            },step=overall_step)
            pbar.set_description(progress_str)
            pbar.update(1)
            if pre_iter%100 == 0:
                print(progress_str)
            if (pre_iter + 1) % args.eval_iters == 0 and args.eval_iters != -1 and pre_iter+1<pretrain_iters_per_round:
                test(val_loader, aggP, loader_type='val', verbose=False)
                test(test_loader, aggP, loader_type='test', verbose=False)
                if not args.no_wandb:
                    wandb.log({"feat_val_acc":aggP['val_stat'][-1]},step=overall_step)
                    wandb.log({"feat_test_acc":aggP['test_stat'][-1]},step=overall_step)
                save_best_rfc_model(model, save_path, aggP, args,rfc=True,rfc_step=last_round_step,rfc_round=round_i)
                model.train()
            if n_iters == pretrain_iters:
                print("Pretrain is done!")
                break

        test(val_loader, aggP, loader_type='val', verbose=False)
        test(test_loader, aggP, loader_type='test', verbose=False)
        if not args.no_wandb:
            wandb.log({"feat_val_acc":aggP['val_stat'][-1]},step=overall_step)
            wandb.log({"feat_test_acc":aggP['test_stat'][-1]},step=overall_step)
        save_best_rfc_model(model, save_path, aggP, args,rfc=True,rfc_step=last_round_step,rfc_round=round_i)
        model.load_state_dict(torch.load(save_path+f"/model_rfc{args.model_name}_r{round_i}_pi{args.pretrain_iters}{'_ls' if args.rfc_long_syn else ''}.rar"))
        
        best_iter = np.argmax(aggP['val_stat'][last_round_step:])+last_round_step
        print(f"Round {round_i} Ends: Val acc {aggP['val_stat'][best_iter]} Test Acc {aggP['test_stat'][best_iter-1]}")
        
        if round_i<args.pretrain_rounds:
            # eval and augment new groups
            model.eval()
            correct_pred = []
            preds = []
            with torch.no_grad():
                for data in tqdm.tqdm(feat_train_loader):
                    x, y = unpack_data(data, device)
                    y_hat = model(x).argmax(-1)
                    correct_pred += (y_hat==y).cpu().tolist()
                    preds += y_hat.cpu().tolist()
            
            preds = [torch.LongTensor(preds)]
            train_loader.dataset.extend_rfc_labels(preds)
            correct_pred = torch.tensor(correct_pred).bool()
            new_feat_domains = [torch.nonzero(correct_pred,as_tuple=True)[0],\
                                torch.nonzero(~correct_pred,as_tuple=True)[0]]
            train_loader.dataset.extend_feat_domains(new_feat_domains)
            train_loader.dataset.reset_feat_batch()
            model.classifier.eval()
            classifiers.append(model.classifier)
    pretrain_iters_per_round = pretrain_iters
    if args.rfc_long_syn:
        pretrain_iters_per_round = pretrain_iters*2
    train_loader.dataset.prepare_rfc_domains()
    train_loader.dataset.reset_feat_batch()
    torch.cuda.empty_cache()
    for syn_round in range(1):
        # re-initialize a model at each RFC round
        model = modelC(args, weights=None).to(device)
        hidden_dim = model.classifier.weight.size(1)
        num_classes = model.classifier.weight.size(0)
        
        last_round_step = len(aggP['val_stat'])
        clf_params = []
        for clf in classifiers:
            clf_params += list(clf.parameters())
        optimizer = opt(list(model.parameters())+clf_params, \
                            **args.optimiser_args)
        
        if args.scheduler is not None and len(args.scheduler)>0:
            scheduler = initialize_scheduler(args, optimizer, n_train_steps)
        else:
            scheduler = None
        
        pbar.reset()
        num_feat_domains = len(train_loader.dataset.feat_domains)
        for pre_iter in range(pretrain_iters_per_round):
            if sum([l >= 1 for l in train_loader.dataset.feat_batches_left.values()]) < pretrain_rounds:
                train_loader.dataset.reset_feat_batch()
                if scheduler is not None and not scheduler.step_every_batch:
                    scheduler.step()
            model.train()
            overall_step += 1
            domains = list(range(num_feat_domains))
            losses_aug = []
            losses_retain = []
            avg_loss = 0
            for (di,domain) in enumerate(domains):
                data = train_loader.dataset.get_feat_batch(domain,rfc_label=True)
                x, y = unpack_data(data, device)
                y_hat, feat = model(x,get_feat=True)

                loss_aug = F.cross_entropy(y_hat,y,reduction="none")
                losses_aug.append(torch.zeros(1).to(device))
                avg_loss += torch.zeros(1).to(device)
                loss_retain = F.cross_entropy(classifiers[domain](feat),y,reduction="none")
                losses_retain.append(loss_retain.mean())
            loss = max(losses_aug).view(1)  # only have the aug loss in RFC
            if len(losses_retain)>0:
                retain_loss = torch.stack(losses_retain).mean()
            else:
                retain_loss = torch.zeros(1).to(device)
            loss += retain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler.step_every_batch:
                scheduler.step()

            # log the number of batches left for each domain
            for domain in domains:
                train_loader.dataset.feat_batches_left[domain] -= 1
            n_iters += 1
            # display progress
            progress_str = f"SynRound Pretrain {pre_iter}/{pretrain_iters_per_round}"+ \
                                    f"iters: Aug loss {max(losses_aug).item():0.5e} Retain loss {retain_loss.item():0.5e}"
            if not args.no_wandb:
                wandb.log({ "aug_loss": max(losses_aug).item(),
                            "retain_loss": retain_loss.item()/(args.retain_penalty+1e-6),
                            },step=overall_step)
            pbar.set_description(progress_str)
            pbar.update(1)
            if pre_iter%100 == 0:
                print(progress_str)
            if (pre_iter + 1) % args.eval_iters == 0 and args.eval_iters != -1 and pre_iter+1<pretrain_iters_per_round:
                model = agg_rfc_weights(model,classifiers,hidden_dim,num_classes)
                test(val_loader, aggP, loader_type='val', verbose=False)
                test(test_loader, aggP, loader_type='test', verbose=False)
                if not args.no_wandb:
                    wandb.log({"feat_val_acc":aggP['val_stat'][-1]},step=overall_step)
                    wandb.log({"feat_test_acc":aggP['test_stat'][-1]},step=overall_step)
                save_best_rfc_model(model, save_path, aggP, args,rfc=True,rfc_step=last_round_step,rfc_round=pretrain_rounds+1)
                model.train()
            if n_iters == pretrain_iters:
                print("Pretrain is done!")
                break
    model = agg_rfc_weights(model,classifiers,hidden_dim,num_classes)
    test(val_loader, aggP, loader_type='val', verbose=False)
    test(test_loader, aggP, loader_type='test', verbose=False)
    if not args.no_wandb:
        wandb.log({"feat_val_acc":aggP['val_stat'][-1]},step=overall_step)
        wandb.log({"feat_test_acc":aggP['test_stat'][-1]},step=overall_step)
    save_best_rfc_model(model, save_path, aggP, args,rfc=True,rfc_step=last_round_step,rfc_round=pretrain_rounds+1)
    best_iter = np.argmax(aggP['val_stat'][last_round_step:])+last_round_step
    print(f"Round {round_i} Ends: Val acc {aggP['val_stat'][best_iter]} Test Acc {aggP['test_stat'][best_iter-1]}")
        
    set_seed(args.seed)
    train_loader, tv_loaders, dataset = modelC.getDataLoaders(args, device=device)
    model = modelC(args, weights=None).to(device)
    model.load_state_dict(torch.load(save_path + f"/model_rfc{args.model_name}_r{args.pretrain_rounds+1}_pi{args.pretrain_iters}{'_ls' if args.rfc_long_syn else ''}.rar"))
    print('Finished feat pre-training!')
    classifier = model.classifier
    trainable_params = classifier.parameters() if args.frozen else model.parameters()
    optimizer = opt([{'params':list(trainable_params),'initial_lr':args.optimiser_args['lr']}], **args.optimiser_args)
    if args.scheduler is not None and len(args.scheduler)>0:
        scheduler = initialize_scheduler(args, optimizer, n_train_steps,last_epoch=overall_step-1)
    else:
        scheduler = None
    overall_step = 0
    pbar.close()
    train_loader.dataset.clean_feat_domains()

if __name__ == '__main__':
    try:
        if args.need_pretrain and args.pretrain_iters != 0:
            pretrain_path = os.path.join(args.exp_dir,"experiments",args.dataset,str(args.seed))
            if not os.path.exists(pretrain_path):
                os.makedirs(pretrain_path)
            if args.use_old:
                model.load_state_dict(torch.load(pretrain_path + f'/model'+(f"_pi{args.pretrain_iters}.rar" if new_pretrain_iters else ".rar")))
                print(f"Load pretrained model from {pretrain_path}+{f'/model'+(f'_pi{args.pretrain_iters}.rar' if new_pretrain_iters else '.rar')}")
            else:
                print("="*30 + "ERM pretraining" + "="*30)
                pretrain(train_loader, args.pretrain_iters, save_path=pretrain_path)
        elif args.need_ifeat_pretrain and args.pretrain_rounds != 0:
            pretrain_path = os.path.join(args.exp_dir,"experiments",args.dataset,str(args.seed))
            if not os.path.exists(pretrain_path):
                os.makedirs(pretrain_path)
            if args.use_old and args.load_feat_round>0:
                pretrain_path = pretrain_path+ f"/model_ifeat{args.model_name}_round_{args.load_feat_round}_rp{args.retain_penalty}_pi{args.pretrain_iters}.rar"
                model.load_state_dict(torch.load(pretrain_path))
                print(f"Load feat pretrained model from {pretrain_path}")
            elif args.use_old:
                pretrain_path = pretrain_path+ f"/model_ifeat{args.model_name}_rp{args.retain_penalty}_pi{args.pretrain_iters}.rar"
                model.load_state_dict(torch.load(pretrain_path ))
                print(f"Load feat pretrained model from {pretrain_path}")
            else:
                print("="*30 + "ifeat pretraining" + "="*30)
                ifeat_train(train_loader, args.pretrain_iters, args.pretrain_rounds, save_path=pretrain_path)
        elif args.need_rfc_pretrain:
            if args.use_old:
                if len(args.rfc_ckpt_path)>0:
                    ckpt_dict = torch.load(args.rfc_ckpt_path)
                    ckpt_dict = ckpt_dict['algorithm']
                    new_ckpt_dict = {}
                    num_classes = model.classifier.weight.size(0)
                    for k in ckpt_dict.keys():
                        if "model.features" in k:
                            new_ckpt_dict[k.replace("model.features.","enc.")] = ckpt_dict[k]
                        elif "model.classifier" in k:
                            overall_dims = ckpt_dict[k].size(0)
                            num_rfc_rounds = overall_dims // num_classes
                            assert overall_dims % num_classes == 0
                            new_ckpt_dict[k[len("model."):]] = torch.stack(torch.split(ckpt_dict[k],num_rfc_rounds)).mean(dim=0)
                        else:
                            new_ckpt_dict[k] = ckpt_dict[k]
                    ckpt_dict = new_ckpt_dict
                    model.load_state_dict(ckpt_dict)
                    print(f"Load RFC pretrained model from {args.rfc_ckpt_path}")
                else:
                    pretrain_path = os.path.join(args.exp_dir,"experiments",args.dataset,str(args.seed))
                    pretrain_path = pretrain_path + f"/model_rfc{args.model_name}_r{args.pretrain_rounds+1}_pi{args.pretrain_iters}{'_ls' if args.rfc_long_syn else ''}.rar"
                    model.load_state_dict(torch.load(pretrain_path))
                    print(f"Load feat pretrained model from {pretrain_path}")
            else:
                pretrain_path = os.path.join(args.exp_dir,"experiments",args.dataset,str(args.seed))
                print("="*30 + "RFC pretraining" + "="*30)
                rfc_train(train_loader, args.pretrain_iters, args.pretrain_rounds, save_path=pretrain_path)
        
        torch.cuda.empty_cache()
        print("="*30 + f"Training: {args.algorithm}" + "="*30)
        train = locals()[f'train_{args.algorithm}']
        agg = defaultdict(list)
        agg['val_stat'] = [0.]

        for epoch in range(args.epochs):
            train(train_loader, epoch, agg)
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg, args)
            if not args.no_wandb:
                wandb.log({"val_acc":agg['val_stat'][-1]},step=overall_step)
                wandb.log({"test_acc":agg['test_stat'][-1]},step=overall_step)
            if scheduler is not None and not scheduler.step_every_batch:
                scheduler.step()
        model.load_state_dict(torch.load(runPath + '/model.rar'))
        print('Finished training! Loading best model...')
        test_acc = 0
        for split, loader in tv_loaders.items():
            tmp_acc = test(loader, agg, loader_type=split, save_ypred=True,return_last=True)
            if split=="test":
                test_acc = tmp_acc
        torch.save(agg,os.path.join(runPath,f"{exp_name}_agg.pt"))
        if not args.no_wandb:
            wandb.finish()
    except Exception as e:
        traceback.print_exc()
        print(e)
        if not args.no_wandb:
            wandb.finish(-1)
            print("Exceptions found, delete all wandb files")
            import shutil
            shutil.rmtree(wandb_run.dir.replace("/files",""))
