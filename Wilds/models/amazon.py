# import os
# from copy import deepcopy

# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from transformers import DistilBertForSequenceClassification
# from transformers import DistilBertTokenizerFast
# from transformers import logging
# from wilds.common.data_loaders import get_eval_loader
# from wilds.datasets.amazon_dataset import AmazonDataset

# from .datasets import GeneralWilds_Batched_Dataset

# logging.set_verbosity_error()

# MAX_TOKEN_LENGTH = 512
# NUM_CLASSES = 5

# def initialize_bert_transform(root_dir="../data"):
#     """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
#     try:
#         tokenizer = DistilBertTokenizerFast.from_pretrained(root_dir)
#     except Exception as e:
#         tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
#         tokenizer.save_pretrained(root_dir)
#     def transform(text):
#         tokens = tokenizer(
#             text,
#             padding='max_length',
#             truncation=True,
#             max_length=MAX_TOKEN_LENGTH,
#             return_tensors='pt')
#         x = torch.stack(
#             (tokens['input_ids'],
#              tokens['attention_mask']),
#             dim=2)
#         x = torch.squeeze(x, dim=0) # First shape dim is always 1
#         return x
#     return transform


# class DistilBertClassifier(DistilBertForSequenceClassification):
#     def __init__(self, config):
#         super().__init__(config)

#     def __call__(self, x,output_hidden_states=False):
#         input_ids = x[:, :, 0]
#         attention_mask = x[:, :, 1]
#         outputs = super().__call__(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=output_hidden_states,
#         )

#         if output_hidden_states:
#             outputs = outputs
#         else:
#             outputs = outputs[0]
#         return outputs

# class Model(nn.Module):
#     def __init__(self, args, weights):
#         super(Model, self).__init__()
#         self.num_classes = NUM_CLASSES
#         # self.model = DistilBertClassifier.from_pretrained(
#         #     'distilbert-base-uncased',
#         #     num_labels=5,
#         # )
#         try:
#             self.model = DistilBertClassifier.from_pretrained(
#                 os.path.join(args.data_dir, 'wilds',args.dataset),
#                 num_labels=5,
                
#             )
#         except Exception as e:
#             self.model = DistilBertClassifier.from_pretrained(
#                 'distilbert-base-uncased',
#                 num_labels=5,
                
#             )
#             self.model.save_pretrained(os.path.join(args.data_dir, 'wilds',args.dataset))
        
#         if weights is not None:
#             self.load_state_dict(deepcopy(weights))
#         self.classifier = self.model.classifier

#     def reset_weights(self, weights):
#         self.model.load_state_dict(deepcopy(weights))
#         self.classifier = self.model.classifier

#     @staticmethod
#     def getDataLoaders(args, device):
#         dataset = AmazonDataset(root_dir=os.path.join(args.data_dir, 'wilds'), download=True)
#         # get all train data
#         transform = initialize_bert_transform()
#         train_data = dataset.get_subset('train', transform=transform)
#         # separate into subsets by distribution
#         train_sets = GeneralWilds_Batched_Dataset(train_data, args.batch_size, domain_idx=0, drop_last=not args.no_drop_last)
#         # take subset of test and validation, making sure that only labels appeared in train
#         # are included
#         datasets = {}
#         for split in dataset.split_dict:
#             if split != 'train':
#                 datasets[split] = dataset.get_subset(split, transform=transform)

#         # get the loaders
#         kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False} \
#             if device.type == "cuda" else {}
#         train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
#         tv_loaders = {}
#         for split, sep_dataset in datasets.items():
#             tv_loaders[split] = get_eval_loader('standard', sep_dataset, batch_size=256, num_workers=args.num_workers)
#         return train_loaders, tv_loaders, dataset



#     def forward(self, x, get_feat=False,frozen_mode=False):
#         if frozen_mode:
#             self.model.eval()
#             self.classifier.train()
#             with torch.no_grad():
#                 outs = self.model(x,output_hidden_states=True)
#                 pooled_output = outs[1][-1][:, 0]
#                 # pooled_output = hidden_state[:, 0]  # (bs, dim)
#                 pooled_output = self.model.pre_classifier(pooled_output)  # (bs, dim)
#                 pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
#                 pooled_output = self.model.dropout(pooled_output)  # (bs, dim)
#             outs = self.classifier(pooled_output)
#             return outs
#         if get_feat:
#             # print(self.model)
#             outs = self.model(x,output_hidden_states=True)
#             with torch.no_grad():
#                 pooled_output = outs[1][-1][:, 0]
#                 # pooled_output = hidden_state[:, 0]  # (bs, dim)
#                 pooled_output = self.model.pre_classifier(pooled_output)  # (bs, dim)
#                 pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
#                 pooled_output = self.model.dropout(pooled_output)  # (bs, dim)
#                 # print(pooled_output.size())
            
#             # print(self.model.classifier)
#             # print(outs)
#             # exit()
#             return outs[0],pooled_output
#         return self.model(x)
import os
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast
from transformers import DistilBertTokenizerFast
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.amazon_dataset import AmazonDataset
from transformers import DistilBertForSequenceClassification, DistilBertModel



class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        

    def __call__(self, x,output_hidden_states=False):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        if output_hidden_states:
            outputs = outputs
        else:
            outputs = outputs[0]
        return outputs


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output

from .datasets import GeneralWilds_Batched_Dataset

# logging.set_verbosity_error()

MAX_TOKEN_LENGTH = 512
NUM_CLASSES = 5

def initialize_bert_transform():
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt')
        x = torch.stack(
            (tokens['input_ids'],
             tokens['attention_mask']),
            dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform



class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES

        try:
            self.model = DistilBertClassifier.from_pretrained(
                os.path.join(args.data_dir, 'wilds',args.dataset),
                num_labels=5,
            )
        except Exception as e:
            self.model = DistilBertClassifier.from_pretrained(
                'distilbert-base-uncased',
                num_labels=5,
            )
            self.model.save_pretrained(os.path.join(args.data_dir, 'wilds',args.dataset))
        # self.model = DistilBertClassifier.from_pretrained(
        #     'distilbert-base-uncased',
        #     num_labels=2,
        #     cache_dir=os.path.join(args.data_dir, 'wilds',args.dataset)
        # )
        if weights is not None:
            self.load_state_dict(deepcopy(weights))
        hidden_dim = self.model.classifier.weight.size(1)
        num_classes = self.model.classifier.weight.size(0)
        self.classifier = nn.Linear(hidden_dim,num_classes)
        self.classifier.weight = nn.Parameter(self.model.classifier.weight)
        self.classifier.bias = nn.Parameter(self.model.classifier.bias)
        self.group_weights = None
    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))
    # def reset_weights(self, weights):
    #     self.model.load_state_dict(deepcopy(weights))
    #     self.classifier = self.model.classifier
    def getDataLoaders(args, device):
        dataset = AmazonDataset(root_dir=os.path.join(args.data_dir, 'wilds'), download=True)
        # get all train data
        transform = initialize_bert_transform()
        train_data = dataset.get_subset('train', transform=transform)
        # separate into subsets by distribution
        train_sets = GeneralWilds_Batched_Dataset(train_data, args.batch_size, domain_idx=0, drop_last=not args.no_drop_last)
        # take subset of test and validation, making sure that only labels appeared in train
        # are included
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)

        # get the loaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        tv_loaders = {}
        for split, sep_dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', sep_dataset, batch_size=256, num_workers=args.num_workers)
        return train_loaders, tv_loaders, dataset



    def forward(self, x, get_feat=False,frozen_mode=False):
        if frozen_mode:
            # pass
            self.model.eval()
            self.classifier.train()
            with torch.no_grad():
                outs = self.model(x,output_hidden_states=True)
                pooled_output = outs[1][-1][:, 0]
                # pooled_output = hidden_state[:, 0]  # (bs, dim)
                pooled_output = self.model.pre_classifier(pooled_output)  # (bs, dim)
                pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
                pooled_output = self.model.dropout(pooled_output)  # (bs, dim)
            outs = self.classifier(pooled_output)
            return outs
        outs = self.model(x,output_hidden_states=True)
        # with torch.no_grad():
        pooled_output = outs[1][-1][:, 0]
        # pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.model.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.model.dropout(pooled_output)  # (bs, dim)
        outs = self.classifier(pooled_output)
        if get_feat:
            return outs,pooled_output
        return outs
