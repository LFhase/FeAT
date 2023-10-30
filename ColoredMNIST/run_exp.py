import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import torch.nn.functional as F
import copy
import os 
from backpack import backpack, extend
from backpack.extensions import BatchGrad

from collections import OrderedDict

from mydatasets import coloredmnist

from models import MLP, TopMLP

from utils import pretty_print, correct_pred,GeneralizedCELoss, EMA, mean_weight, mean_nll, mean_mse, mean_accuracy,validation, parse_bool

from train import get_train_func



def main(flags):
    if flags.save_dir is not None and not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)
    flags.freeze_featurizer = False if flags.freeze_featurizer.lower() == 'false' else True 
    final_train_accs = []
    final_train_losses = []
    final_test_accs = []
    final_test_losses = []
    logs = []


    for restart in range(flags.n_restarts):

        if flags.verbose:
            print("Restart", restart)
        

        ### loss function binary_cross_entropy 
        input_dim = 2 * 14 * 14
        if flags.methods in ['rsc', 'lff']:
            n_targets = 2
            lossf = F.cross_entropy
            int_target = True 
        else:
            n_targets = 1 
            lossf = mean_nll 
            int_target = False  


        np.random.seed(restart)
        torch.manual_seed(restart)
        ### load datasets 
        if flags.dataset == 'coloredmnist025':
            envs, test_envs = coloredmnist(0.25, 0.1, 0.2, int_target = int_target)
        elif flags.dataset == 'coloredmnist025gray':
            envs, test_envs = coloredmnist(0.25, 0.5, 0.5,int_target = int_target)
        elif flags.dataset == 'coloredmnist01':
            envs, test_envs = coloredmnist(0.1, 0.2, 0.25, int_target = int_target)
        elif flags.dataset == 'coloredmnist01gray':
            envs, test_envs = coloredmnist(0.1, 0.5, 0.5,  int_target = int_target)
        else:
            raise NotImplementedError
        

        mlp = MLP(hidden_dim = flags.hidden_dim, input_dim=input_dim).cuda()
        topmlp = TopMLP(hidden_dim = flags.hidden_dim, n_top_layers=flags.n_top_layers, \
            n_targets=n_targets, fishr= flags.methods=='fishr').cuda()

        print(mlp, topmlp)

        if flags.load_model_dir is not None and os.path.exists(flags.load_model_dir):
            device = torch.device("cuda")
            state = torch.load(os.path.join(flags.load_model_dir,'mlp%d.pth' % restart), map_location=device)
            mlp.load_state_dict(state)
            
            state = torch.load(os.path.join(flags.load_model_dir,'topmlp%d.pth' % restart), map_location=device)
            topmlp.load_state_dict(state)
            print("Load model from %s" % flags.load_model_dir)
            

        if len(flags.group_dirs)>0:
            print('load groups')
            x = torch.cat([env['images'] for env in envs])
            y = torch.cat([env['labels'] for env in envs])
            #print(x.shape, y.shape)
            groups = [np.load(os.path.join(group_dir,'group%d.npy' % restart)) for group_dir in flags.group_dirs]
            n_groups = len(groups)
            new_envs = []

            for group in groups:
                for val in np.unique(group):
                    env = {}
                    env['images'] = x[group == val]
                    env['labels'] = y[group == val]

                    new_envs.append(env)
            train_envs = new_envs

        else:
            train_envs = envs

        train_func = get_train_func(flags.methods)
        params = [mlp, topmlp, flags.steps, train_envs, test_envs,lossf,\
            flags.penalty_anneal_iters, flags.penalty_weight, \
            flags.anneal_val, flags.lr, \
            flags.l2_regularizer_weight, flags.freeze_featurizer, flags.eval_steps, flags.verbose, ]
        if flags.methods in ['vrex', 'iga','irm','fishr','gm','lff','erm','dro','ifat','fat']:
            res = train_func(*params)
        elif flags.methods in ['clove']:
            hparams = {'batch_size': flags.batch_size, 'kernel_scale': flags.kernel_scale}
            res = train_func(*params, hparams)
        elif flags.methods in ['rsc']:
            hparams = {'rsc_f_drop_factor' : flags.rsc_f, 'rsc_b_drop_factor': flags.rsc_b}
            res = train_func(*params, hparams)
        elif flags.methods in ['sd']:
            hparams = {'lr_s2_decay': flags.lr_s2_decay}
            res = train_func(*params, hparams)
        else:
            raise NotImplementedError
        hparams['stage2_methods'] = flags.stage2_methods
        hparams['rounds'] = flags.rounds
        hparams['steps_per_round'] = flags.steps_per_round
        (train_acc, train_loss, test_worst_acc, test_worst_loss), per_logs = res 
        
        
        logs.extend(per_logs)
        final_train_accs.append(train_acc)
        final_train_losses.append(train_loss)
        final_test_accs.append(test_worst_acc)
        final_test_losses.append(test_worst_loss)

        if flags.verbose:
            print('Final train acc (mean/std across restarts so far):')
            print(np.mean(final_train_accs), np.std(final_train_accs))
            print('Final train loss (mean/std across restarts so far):')
            print(np.mean(final_train_losses), np.std(final_train_losses))
            print('Final worest test acc (mean/std across restarts so far):')
            print(np.mean(final_test_accs), np.std(final_test_accs))
            print('Final worest test loss (mean/std across restarts so far):')
            print(np.mean(final_test_losses), np.std(final_test_losses))

        results = [np.mean(final_train_accs), np.std(final_train_accs), 
                                np.mean(final_train_losses), np.std(final_train_losses), 
                                np.mean(final_test_accs), np.std(final_test_accs), 
                                np.mean(final_test_losses), np.std(final_test_losses), 
                                ]
        
    

        if flags.save_dir is not None:
            state = mlp.state_dict()
            torch.save(state, os.path.join(flags.save_dir,'mlp%d.pth' % restart))
            state = topmlp.state_dict()
            torch.save(state, os.path.join(flags.save_dir,'topmlp%d.pth'  % restart))
            
            with torch.no_grad():
                x = torch.cat([env['images'] for env in envs])
                y = torch.cat([env['labels'] for env in envs])
                logits = topmlp(mlp(x))
            group, _ = correct_pred(logits, y)

            pseudolabel = np.copy(y.cpu().numpy().flatten())
            pseudolabel[~group] = 1-pseudolabel[~group]
            np.save(os.path.join(flags.save_dir,'group%d.npy' % restart), group)
            np.save(os.path.join(flags.save_dir,'pseudolabel%d.npy' % restart), pseudolabel )

    logs = np.array(logs)
    
    if flags.save_dir is not None:
        np.save(os.path.join(flags.save_dir,'logs.npy'), logs)

    return results, logs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Colored MNIST & CowCamel')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='coloredmnist025')
    parser.add_argument('--hidden_dim', type=int, default=390)
    parser.add_argument('--n_top_layers', type=int, default=1)
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.0011)
    
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--steps', type=int, default=501)
    parser.add_argument('--lossf', type=str, default='nll')
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--anneal_val', type=float, default=1)
    
    parser.add_argument('--methods', type=str, default='irmv2')
    parser.add_argument('-s2','--stage2_methods', type=str, default='irm')
    parser.add_argument('-r','--rounds', type=int,default=2)
    parser.add_argument('-sr','--steps_per_round', type=int,default=151)
    parser.add_argument('--lr_s2_decay', type=float, default=500)
    parser.add_argument('--freeze_featurizer', type=str, default='False')
    parser.add_argument('--eval_steps', type=int, default=5)
    
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--group_dirs', type=str, nargs='*',default={})
    
    #RSC
    parser.add_argument('--rsc_f', type=float, default=0.99)
    parser.add_argument('--rsc_b', type=float, default=0.97)

    #clove 
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--kernel_scale', type=float, default=0.4)

    parser.add_argument('--n_examples', type=int, default=18000)

    parser.add_argument('--norun',type=parse_bool, default=False)
    flags = parser.parse_args()

    if flags.norun:
        if flags.verbose:
            print('Flags:')
            for k,v in sorted(vars(flags).items()):
                print("\t{}: {}".format(k, v))
    else:   
        main(flags)





