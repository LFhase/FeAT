import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import tqdm

class LogisticRegression:
    def __init__(self,penalty,C,solver):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.coef_ = None
        self.intercept_ = None
        self.lin1 = None


    def fit(self, X, Y, lr=0.01, iterations=10000):
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        device = torch.device("cuda")
        input_dim = X.size(1)
        output_dim = Y.max()+1
        if self.lin1 is None:
            self.lin1 = nn.Linear(input_dim,output_dim).to(device)
            nn.init.xavier_uniform_(self.lin1.weight)
            nn.init.zeros_(self.lin1.bias)
            print(self.lin1)
        self.lin1.train()
        X = X.to(device)
        Y = Y.to(device)
        optimizer = torch.optim.SGD(self.lin1.parameters(),lr=lr)

        # for i in range(iterations):
        best_train_acc = 0
        early_stop_epoch = 0
        for ep in (pbar:= tqdm.tqdm(range(iterations))):
            new_idx = torch.randperm(X.size(0))
            X = X[new_idx]
            Y = Y[new_idx]
            Y_pred = self.lin1(X)
            loss = F.cross_entropy(Y_pred,Y)
            train_acc = (Y_pred.argmax(-1)==Y).sum()/Y.size(0)
            # loss = F.mse_loss(Y_pred,Y)
            # print(f"Epoch {i}:{loss.item()}")
            pbar.set_description("Epoch {:5d}: Loss={:.3f} Acc={:3f}".format(
            ep,loss.item(),train_acc.item()))
            if self.penalty.lower() == 'l1':
                for w in self.lin1.parameters():
                    loss += (1/self.C)*w.abs().mean()
            elif self.penalty.lower() == 'l2':
                for w in self.lin1.parameters():
                    loss += (1/self.C)*w.norm().pow(2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                early_stop_epoch = 0
            elif ep > 1000:
                early_stop_epoch += 1
                if early_stop_epoch >= 100:
                    print("Early stopped")
                    break
        self.coef_ = self.lin1.weight.clone().detach()
        self.intercept_ = self.lin1.bias.clone().detach()
        return self
    
    def reset(self):
        if self.coef_ is not None and self.intercept_ is not None:
            print("Reset")
            self.lin1.weight = nn.Parameter(self.coef_.clone())
            self.lin1.bias = nn.Parameter(self.intercept_.clone())
            self.intercept_ = None
            self.coef_ = None

    def predict(self, X):
        
        with torch.no_grad():
            X = torch.from_numpy(X)
            self.lin1.eval()
            X = X.cuda()
            return self.lin1(X).argmax(-1).cpu().numpy()
