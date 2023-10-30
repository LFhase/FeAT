# modified from https://github.com/YujiaBao/Predict-then-Interpolate/blob/main/src/main.py
import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import torch.nn.functional as F
import copy
import os
from mydatasets import coloredmnist
from models import MLP, TopMLP
from utils import pretty_print, correct_pred,GeneralizedCELoss, EMA, mean_weight, mean_nll, mean_mse, mean_accuracy

def validation(model, envs, test_envs, lossf): 
	with torch.no_grad():
		for env in envs + test_envs:
			logits = model(env['images'])

			env['nll'] = lossf(logits, env['labels'])
			env['acc'] = mean_accuracy(logits, env['labels'])

	test_worst_loss = torch.stack([env['nll'] for env in test_envs]).max()
	test_worst_acc  = torch.stack([env['acc'] for env in test_envs]).min()
	train_loss = torch.stack([env['nll'] for env in envs]).mean()
	train_acc  = torch.stack([env['acc'] for env in envs]).mean()
	
	return train_loss.detach().cpu().numpy(), train_acc.detach().cpu().numpy(), \
	test_worst_loss.detach().cpu().numpy(),test_worst_acc.detach().cpu().numpy()
	
parser = argparse.ArgumentParser(description='Colored MNIST & CowCamel')
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--dataset', type=str, default='coloredmnist025')
parser.add_argument('--hidden_dim', type=int, default=390)
parser.add_argument('--n_top_layers', type=int, default=2)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.0011)
parser.add_argument('--lr', type=float, default=0.0005 )
parser.add_argument('--steps1', type=int, default=51)
parser.add_argument('--steps3', type=int, default=701)
parser.add_argument('--lossf', type=str, default='nll')
parser.add_argument('--save_dir', type=str, default='.')

flags = parser.parse_args()
if flags.dataset == 'coloredmnist025':
	envs, test_envs = coloredmnist(0.25, 0.1, 0.2, int_target = False)
elif flags.dataset == 'coloredmnist01':
	envs, test_envs = coloredmnist(0.1, 0.2, 0.25, int_target = False)
logs = []
for step in range(flags.n_restarts):
	## load datasets 
	num_envs = len(envs)            
	## init models 
	input_dim = 14*14*2
	n_targets = 1 
	models = []
	def get_topmlp_func():
		return TopMLP(hidden_dim = flags.hidden_dim,  n_top_layers=flags.n_top_layers, n_targets=n_targets).cuda()

	for i in range(num_envs):
		mlp = MLP(hidden_dim = flags.hidden_dim, input_dim=input_dim).cuda()
		topmlp = get_topmlp_func()
		model = torch.nn.Sequential(mlp, topmlp)
		models.append(model)


	## Stage1: train models for each env. earlystopping on a 10% validation dataset (controled by --step1).
	for i in range(num_envs):
		print(i)
		
		x, y = envs[i]['images'], envs[i]['labels']
		idx = np.arange(len(x))
		np.random.shuffle(idx)

		val_x, val_y = x[idx[:int(len(idx)*0.1)]], y[idx[:int(len(idx)*0.1)]]
		x, y = x[idx[int(len(idx)*0.1):]], y[idx[int(len(idx)*0.1):]]

		model = models[i]
		optimizer = optim.Adam(model.parameters(), lr=flags.lr)
		lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
		
		for step in range(flags.steps1):
			logits = model(x)
			loss = lossf(logits, y)
			weight_norm = 0
			for w in model.parameters():
				weight_norm += w.norm().pow(2)
			loss += flags.l2_regularizer_weight * weight_norm

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


			if step % 50 == 0:
				with torch.no_grad():
					tr_acc = mean_accuracy(model(x),y)
					val_acc = mean_accuracy(model(val_x),val_y)
					#$print(tr_acc, val_acc)
					pretty_print(np.int32(step),tr_acc.detach().cpu().numpy(), val_acc.detach().cpu().numpy())
				# train_loss, train_acc, test_worst_loss, test_worst_acc = \
				# validation(model, [envs[i]], test_envs, lossf)
				# log = [np.int32(step), train_loss, train_acc,test_worst_loss, test_worst_acc]
				# if flags.verbose:
				# 	pretty_print(*log)


	## Stage2: cross group splitting
	created_envs = [] 
	for i in range(num_envs):
		
		x, y = envs[i]['images'], envs[i]['labels']
		model = models[i] 
		model.eval()
		pred_y = model(x)
		correct, uncorrect = correct_pred(pred_y,y)
		created_envs.append( {'images':x[correct], 'labels':y[correct]})
		created_envs.append( {'images':x[uncorrect], 'labels':y[uncorrect]})

	## Stage3: 	DRO training
	### init model
	mlp = MLP(hidden_dim = flags.hidden_dim, input_dim=input_dim).cuda()
	topmlp = get_topmlp_func()
	model = torch.nn.Sequential(mlp, topmlp)
	optimizer = optim.Adam(model.parameters(), lr=flags.lr)
	lossf = mean_nll if flags.lossf == 'nll' else mean_mse 
	### dro training
	

	for step in range(flags.steps3):
		losses = []
		for env in created_envs:
			x,y = env['images'], env['labels']
			losses.append(lossf(model(x),y))
			#print(losses)
		losses = torch.stack(losses)
		loss = losses[torch.argmax(losses)]

		weight_norm = 0
		for w in model.parameters():
			weight_norm += w.norm().pow(2)
		loss += flags.l2_regularizer_weight * weight_norm

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % 5 == 0:
			train_loss, train_acc, test_worst_loss, test_worst_acc = \
			validation(model, envs, test_envs, lossf)
			log = [np.int32(step), train_loss, train_acc,test_worst_loss, test_worst_acc]
			logs.append(log)
			if flags.verbose:
				pretty_print(*log)
if not os.path.exists(flags.save_dir):
	os.mkdir(flags.save_dir)
np.save(os.path.join(flags.save_dir, '%s_%s_PI_%d.npy' % (flags.dataset,flags.lossf, flags.steps1)), logs)


