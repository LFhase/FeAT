"""Evaluate DFR on spurious correlations datasets."""

import argparse
import random
import torch

import numpy as np
import os
import sys
import tqdm
import json
import pickle

from sklearn.linear_model import LogisticRegression
# from LR import LogisticRegression
from sklearn.preprocessing import StandardScaler

import models
import utils
from utils import supervised_utils

has_wandb = False


has_cuda = torch.cuda.device_count()>0

# WaterBirds
C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
REG = "l1"


import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed)  # canceled as we only use one gpu
# set_seed(0)


def get_args():
    parser = utils.get_model_dataset_args()
    # General
    parser.add_argument(
        "--result_path", type=str, default="logs/",
        help="Path to save results")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, required=False,
        help="Checkpoint path")
    parser.add_argument(
        "--batch_size", type=int, default=100, required=False,
        help="Checkpoint path")
    parser.add_argument('-sb',
        "--save_embeddings", action='store_true',
        help="Save embeddings on disc")
    parser.add_argument(
        "--predict_spurious", action='store_true',
        help="Predict spurious attribute instead of class label")
    parser.add_argument(
        "--drop_group", type=int, default=None, required=False,
        help="Drop group from evaluation")
    parser.add_argument(
        "--seed", type=int, default=0, required=False)
    parser.add_argument(
        "--max_iter", type=int, default=100, required=False)
    parser.add_argument(
        "--log_dir", type=str, default="", help="For loading wandb results")
    parser.add_argument(
        "-ep","--epoch", type=str, default="", help="For logging the specified epoch")
    parser.add_argument(
        "--save_linear_model", action='store_true', help="Save linear model weights")
    parser.add_argument(
        "--save_best_epoch", action='store_true', help="Save best epoch num to pkl")
    # DFR TR
    parser.add_argument(
        "--dfr_train", action='store_true', help="Use train data for reweighting")
    parser.add_argument('-jb',
        "--jumpy_base_eval", action='store_true', help="Use train data for reweighting")
    parser.add_argument('-cheat',
        "--dare_cheating", action='store_true', help="Use train & test data for training")
    parser.add_argument('-nb',
        "--use_new_embeddings", action='store_true', help="Donot use old embeddings")
    parser.add_argument('-ms',
        "--model_selection", type=str, default="wga", help="which way to select best DFR models")
    parser.add_argument('-feat',
        "--eval_feat", action='store_true', help="Whether evaluating feat trained model")
    parser.add_argument('-rfc',
        "--eval_rfc", action='store_true', help="Whether evaluating feat trained model")
    args = parser.parse_args()
    args.num_minority_groups_remove = 0
    args.reweight_groups = False
    args.reweight_spurious = False
    args.reweight_classes = False
    args.no_shuffle_train = True
    args.mixup = False
    args.load_from_checkpoint = True
    if args.seed >= 0:
        set_seed(args.seed)
    return args


def dfr_on_validation_tune(args,
        all_embeddings, all_y, all_g, all_p,preprocess=True, num_retrains=1):

    worst_accs = {}
    mean_accs = {}

    if "iwild" in args.dataset.lower():
        cnt_y = np.bincount(all_y["val"])
        valid_idx = []
        print(len(all_y["val"]),len(all_g["val"]),len(all_embeddings["val"]),len(all_p['val']))
        for (y,cnt) in enumerate(cnt_y):
            if cnt <= 50 and cnt>0:
                print(f"removing class {y} with {cnt} samples")
                valid_idx = all_y["val"]!=y
                all_y["val"] = all_y["val"][valid_idx]
                all_g["val"] = all_g["val"][valid_idx]
                if len(all_y["val"])<len(all_p["val"]):
                    all_p["val"] = all_p["val"][valid_idx]
                all_embeddings["val"] = all_embeddings["val"][valid_idx]
        print("using groups: ", np.bincount(all_g["val"]))
        print("using labels: ", np.bincount(all_y["val"]))
    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1
        
        n_val = len(x_val) // 2
        idx = np.arange(len(x_val))
        np.random.shuffle(idx)

        x_train = x_val[idx[n_val:]]
        y_train = y_val[idx[n_val:]]
        g_train = g_val[idx[n_val:]]

        n_groups = np.max(g_train) + 1
        if "fmow" in args.dataset.lower():
            # drop last group according to the paper
            n_groups -= 1
          
        g_idx = []

        for g in range(n_groups):
            if len(np.where(g_train==g)[0])>(50 if "amazon" in args.dataset.lower() else 100):
                g_idx.append(np.where(g_train==g)[0])

        print([len(g) for g in g_idx])
        min_g = np.min([(len(g) if len(g)>0 else 1e9) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
        y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
        g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])

        x_val = x_val[idx[:n_val]]
        y_val = y_val[idx[:n_val]]
        g_val = g_val[idx[:n_val]]

        print("Val tuning:", np.bincount(g_train))
        if preprocess:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

        for c in C_OPTIONS:
            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",max_iter=args.max_iter)
            logreg.classes = np.arange(np.max(y_train) + 1)
            logreg.fit(x_train, y_train)
            preds_val = logreg.predict(x_val)
            if 'iwild' in args.dataset.lower():
                from sklearn.metrics import f1_score
                group_accs = np.array([f1_score(y_val, preds_val, average='macro')])
            else:
                group_accs = np.array(
                    [(preds_val == y_val)[g_val == g].mean()
                    for g in range(n_groups)])
            worst_acc = np.nanmin(group_accs)
            mean_acc = np.nanmean(group_accs)
            if i == 0:
                worst_accs[c] = worst_acc
                mean_accs[c] = mean_acc
            else:
                worst_accs[c] += worst_acc
                mean_accs[c] += mean_acc
    if args.model_selection.lower() == "avg":
        ks, vs = list(mean_accs.keys()), list(mean_accs.values())
        best_hypers = ks[np.argmax(vs)]
    else:
        ks, vs = list(worst_accs.keys()), list(worst_accs.values())
        best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(
        args, c, all_embeddings, all_y, all_g, target_type="target", num_retrains=20,
        preprocess=True):
    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["val"])
    num_retrials = 50 if "nothing" in args.dataset.lower() else 20
    
    print("using groups: ", np.bincount(all_g["val"]))
    print("using labels: ", np.bincount(all_y["val"]))
    for i in range(num_retrains):
        for _ in range(num_retrials):
            x_val = all_embeddings["val"]
            y_val = all_y["val"]
            g_val = all_g["val"]
            n_groups = np.max(g_val) + 1
            if "fmow" in args.dataset.lower():
                # drop last group according to the paper
                n_groups -= 1
            g_idx = []
            for g in range(n_groups):
                if len(np.where(g_val==g)[0])>(50 if "amazon" in args.dataset.lower() else 100):
                    g_idx.append(np.where(g_val==g)[0])
            min_g = np.min([(len(g) if len(g)>0 else 1e9) for g in g_idx])
            for g in g_idx:
                np.random.shuffle(g)
            x_train = np.concatenate([x_val[g[:min_g]] for g in g_idx])
            y_train = np.concatenate([y_val[g[:min_g]] for g in g_idx])
            g_train = np.concatenate([g_val[g[:min_g]] for g in g_idx])

            if np.any(np.unique(y_train) != np.unique(all_y["val"])):
                # do we need the same thing in tuning?
                print("missing classes, reshuffling...")
                continue
            else:
                break

        print("selected groups: ", np.bincount(g_train))
        print("selected labels: ", np.bincount(y_train))
        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",max_iter=args.max_iter)
        logreg.classes = np.arange(np.max(y_train) + 1)
        logreg.fit(x_train, y_train)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]
    print(np.bincount(g_test))

    if preprocess:
        x_test = scaler.transform(x_test)
    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",max_iter=args.max_iter)
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    print(x_train.shape,n_classes)
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    if torch.is_tensor(coefs[0]):
        logreg.coef_ = torch.stack(coefs).mean(dim=0)
        logreg.intercept_ = torch.stack(intercepts).mean(dim=0)
        logreg.reset()
    else:
        logreg.coef_ = np.mean(coefs, axis=0)
        logreg.intercept_ = np.mean(intercepts, axis=0)

    preds_test = logreg.predict(x_test)
    preds_train = logreg.predict(x_train)
    n_groups = np.max(g_train) + 1
    if 'iwild' in args.dataset.lower():
        from sklearn.metrics import f1_score
        test_accs = np.array([f1_score(y_test, preds_test, average='macro')])
        test_mean_acc = test_accs[0]
        train_accs = np.array([f1_score(y_train, preds_train, average='macro')])
    else:
        test_accs = [(preds_test == y_test)[g_test == g].mean()
                    for g in range(n_groups)]
        test_mean_acc = (preds_test == y_test).mean()
        train_accs = [(preds_train == y_train)[g_train == g].mean()
                    for g in range(n_groups)]
    # remove nan group results
    test_accs = [x for x in test_accs if str(x) != 'nan']
    train_accs = [x for x in train_accs if str(x) != 'nan']


    return test_accs, test_mean_acc, train_accs


def main(args):
    print(args)

    # Load data
    logger = utils.Logger() if not has_wandb else None
    train_loader, test_loader_dict, get_ys_func = (
        utils.get_data(args, logger, contrastive=False))
    n_classes = train_loader.dataset.n_classes
    # Model
    model_datasets = ["fmow","camelyon","civil","amazon","iwildcam","poverty","rxrx"]
    for md in model_datasets:
        if md in args.dataset.lower():
            print("Loading model class ",md)
            model_dataset = md
            break
    modelC = getattr(models, model_dataset)
    model = modelC(args, weights=None).cuda()
    if args.ckpt_path and args.load_from_checkpoint:
        print(f"Loading weights {args.ckpt_path}")
        ckpt_dict = torch.load(args.ckpt_path)
        try:
            model.load_state_dict(ckpt_dict)
        except:
            print("Loading one-output Checkpoint")
            if args.eval_feat:
                w = ckpt_dict["classifier.weight"]
                b = ckpt_dict["classifier.bias"]
                w_ = torch.zeros(w.size())
                b_ = torch.zeros(b.size())
            elif args.eval_rfc:
                ckpt_dict = ckpt_dict['algorithm']

                new_ckpt_dict = {}
                for k in ckpt_dict.keys():
                    if "model.features" in k:
                        new_ckpt_dict[k.replace("model.features.","enc.")] = ckpt_dict[k]
                    elif "model.classifier" in k:
                        new_ckpt_dict[k[len("model."):]] = ckpt_dict[k][:n_classes]
                    else:
                        new_ckpt_dict[k] = ckpt_dict[k]
                ckpt_dict = new_ckpt_dict
            else:
                w = ckpt_dict["fc.weight"]
                b = ckpt_dict["fc.bias"]
                w_ = torch.zeros((2, w.shape[1]))
                w_[1, :] = w
                b_ = torch.zeros((2,))
                b_[1] = b
                ckpt_dict["fc.weight"] = w_
                ckpt_dict["fc.bias"] = b_
            
            
            model.load_state_dict(ckpt_dict)
    else:
        print("Using initial weights")
    print(model)
    model.cuda()
    model.eval()

    # Evaluate model
    if args.jumpy_base_eval:
        base_model_results = {}
    else:
        print("Base Model")
        base_model_results = supervised_utils.eval(model, test_loader_dict)
        base_model_results = {
            name: utils.get_results(accs, get_ys_func) for name, accs in base_model_results.items()}
        print(base_model_results)
        print()
    if args.eval_feat or args.eval_rfc:
        if "civil" in args.dataset.lower():
            model.model.classifier = torch.nn.Identity()
        model.classifier = torch.nn.Identity()
    else:
        model.fc = torch.nn.Identity()
    print(model)
    #splits = ["test", "val"]
    splits = {
        "test": test_loader_dict["test"],
        "val": test_loader_dict["val"]
    }
    if args.dfr_train:
        splits["train"] = train_loader
    elif args.dare_cheating:
        splits["val"] = train_loader
    print(splits.keys())
    if os.path.exists(f"{args.result_path[:-4]}.npz") and not args.use_new_embeddings:
        arr_z = np.load(f"{args.result_path[:-4]}.npz")

        all_embeddings = {split: arr_z[f"embeddings_{split}"] for split in splits}
        all_y = {split: arr_z[f"y_{split}"] for split in splits}
        all_p = {split: arr_z[f"p_{split}"] for split in splits}
        all_g = {split: arr_z[f"g_{split}"] for split in splits}
    else:
        all_embeddings = {}
        all_y, all_p, all_g = {}, {}, {}
        for name, loader in splits.items():
            all_embeddings[name] = []
            all_y[name], all_p[name], all_g[name] = [], [], []
            # print(name,loader)
            for x, y, g, p in tqdm.tqdm(loader):
                with torch.no_grad():
                    all_embeddings[name].append(model(x.cuda()).detach().cpu().numpy())
                    all_y[name].append(y.detach().cpu().numpy())
                    all_g[name].append(g.detach().cpu().numpy())
                    all_p[name].append(p.detach().cpu().numpy())
            all_embeddings[name] = np.vstack(all_embeddings[name])
            all_y[name] = np.concatenate(all_y[name])
            all_g[name] = np.concatenate(all_g[name])
            all_p[name] = np.concatenate(all_p[name])
        # print(all_g[name].max())
        if args.save_embeddings:
            np.savez(f"{args.result_path[:-4]}.npz",
                     embeddings_test=all_embeddings["test"],
                     embeddings_val=all_embeddings["val"],
                     y_test=all_y["test"],
                     y_val=all_y["val"],
                     g_test=all_g["test"],
                     g_val=all_g["val"],
                     p_test=all_p["test"],
                     p_val=all_p["val"],
                    )

    if args.drop_group is not None:
        print("Dropping group", args.drop_group)
        print([name for name in splits])
        all_masks = {name: all_g[name] != args.drop_group for name in splits}
        for name in splits:
            all_y[name] = all_y[name][all_masks[name]]
            all_g[name] = all_g[name][all_masks[name]]
            all_p[name] = all_p[name][all_masks[name]]
            all_embeddings[name] = all_embeddings[name][all_masks[name]]
    
    if args.dfr_train:
        print("Reweighting on training data")
        all_y["val"] = all_y["train"]
        all_g["val"] = all_g["train"]
        all_p["val"] = all_p["train"]
        all_embeddings["val"] = all_embeddings["train"]

    if args.dare_cheating:
        print("Cheating via usinge train & test data")
        all_y["val"] = np.concatenate((all_y["val"],all_y["test"]))
        # del all_y["train"]
        all_g["val"] = np.concatenate((all_g["val"],all_g["test"]))
        # del all_g["train"]
        all_p["val"] = np.concatenate((all_p["val"],all_p["test"]))
        # del all_p["train"]
        all_embeddings["val"] = np.concatenate((all_embeddings["val"],all_embeddings["test"]))
        # del all_embeddings["train"]


    # DFR on validation
    print("DFR")
    dfr_results = {}
    c = dfr_on_validation_tune(args,all_embeddings, all_y, all_g,all_p)
    dfr_results["best_hypers"] = c
    print("Hypers:", (c))
    test_accs, test_mean_acc, train_accs = dfr_on_validation_eval(
        args, c, all_embeddings, all_y, all_g, target_type="target")
    dfr_results["test_accs"] = test_accs
    dfr_results["train_accs"] = train_accs
    dfr_results["test_worst_acc"] = np.min(test_accs)
    dfr_results["test_mean_acc"] = test_mean_acc
    print(dfr_results)
    print()

    all_results = {}
    if len(args.epoch)>0:
        all_results["epoch"] = args.epoch
    all_results[f"base_model_results"] = base_model_results
    all_results[f"dfr_val_results"] = dfr_results

    if args.predict_spurious:
        print("Predicting spurious attribute")
        all_y = all_p

        # DFR on validation
        print("DFR (spurious)")
        dfr_spurious_results = {}
        c = dfr_on_validation_tune(args,
            all_embeddings, all_y, all_g, all_p)
        dfr_spurious_results["best_hypers"] = c
        print("Hypers:", (c))
        test_accs, test_mean_acc, train_accs = dfr_on_validation_eval(
            args, c, all_embeddings, all_y, all_g, target_type="spurious")
        dfr_spurious_results["test_accs"] = test_accs
        dfr_spurious_results["train_accs"] = train_accs
        dfr_spurious_results["test_worst_acc"] = np.min(test_accs)
        dfr_spurious_results["test_mean_acc"] = test_mean_acc
        print(json.dumps(dfr_spurious_results))
        print()

        all_results[f"dfr_val_spurious_results"] = dfr_spurious_results
    
    print(json.dumps(all_results))


    command = " ".join(sys.argv)
    all_results["command"] = command
    all_results["model"] = args.model

    if args.ckpt_path:
        if os.path.exists(os.path.join(os.path.dirname(args.ckpt_path), 'args.json')):
            base_model_args_file = os.path.join(os.path.dirname(args.ckpt_path), 'args.json')
            with open(base_model_args_file) as fargs:
                base_model_args = json.load(fargs)
                all_results["base_args"] = base_model_args
        if args.save_best_epoch:
            if os.path.exists(os.path.join(os.path.dirname(args.ckpt_path), 'best_epoch_num.npy')):
                base_epoch_file = os.path.join(os.path.dirname(args.ckpt_path), 'best_epoch_num.npy')
                best_epoch_num = np.load(base_epoch_file)[0]
                all_results["base_model_best_epoch"] = best_epoch_num

    with open(args.result_path, 'wb') as f:
        pickle.dump(all_results, f)

    if has_wandb:
        wandb.log(all_results)


if __name__ == '__main__':
    args = get_args()
    main(args)
