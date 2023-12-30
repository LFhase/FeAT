import copy
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Poverty_Batched_Dataset(Dataset):
    """
    Batched dataset for Poverty. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, dataset, split, batch_size, transform=None, drop_last=True):
        self.split_array = dataset.split_array
        self.split_dict = dataset.split_dict
        split_mask = self.split_array == self.split_dict[split]
        self.split_idx = np.where(split_mask)[0]

        self.root = dataset.root
        self.no_nl = dataset.no_nl

        self.metadata_array = torch.stack([dataset.metadata_array[self.split_idx, i] for i in [0, 2]], -1)
        self.y_array = dataset.y_array[self.split_idx]

        self.eval = dataset.eval
        self.collate = dataset.collate
        self.metadata_fields = dataset.metadata_fields
        self.data_dir = dataset.data_dir

        self.transform = transform if transform is not None else lambda x: x

        domains = self.metadata_array[:, 1]
        self.domain_indices = [torch.nonzero(domains == loc).squeeze(-1)
                               for loc in domains.unique()]
        self.num_envs = len(domains.unique())
        for did in self.domain_indices:
            print(len(did))
        self.domains = domains
        self.targets = self.y_array
        self.batch_size = batch_size
        self.drop_last = drop_last

        min_domain_size = np.min([len(didx) for didx in self.domain_indices])
        self.training_steps = int(min_domain_size/self.batch_size)+\
                                (not self.drop_last*(min_domain_size//self.batch_size>0))
        self.feat_domains = [torch.arange(self.y_array.size(0))] # domain 0
        self.rfc_labels = [] # domain 0
    
    def get_feat_batch(self,feat_domain,rfc_label=False):
        batch_index = self.feat_batch_indices[feat_domain][len(self.feat_batch_indices[feat_domain]) - self.feat_batches_left[feat_domain]]
        domains = torch.zeros(batch_index.size()).to(batch_index.device)
        domains += feat_domain
        if rfc_label:
            if len(self.rfc_labels)==len(self.feat_domains):
                targets = self.rfc_labels[feat_domain][batch_index]
            else:
                targets = self.rfc_labels[feat_domain//2][batch_index]
            return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               targets, domains
        else:
            return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
                self.targets[batch_index], domains
    
    def reset_feat_batch(self,train_loader_iter=None):
        self.train_loader_iter = train_loader_iter
        self.feat_batch_indices, self.feat_batches_left = {}, {}
        if self.train_loader_iter is not None:
            assert len(self.feat_domains)==1
            self.feat_batches_left[0] = len(self.train_loader_iter)
            self.feat_batch_indices[0] = torch.arange(len(self.train_loader_iter))
        else:
            for loc, d_idx in enumerate(self.feat_domains):
                self.feat_batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
                # mannually drop last
                if self.drop_last and len(self.feat_batch_indices[loc][-1])<self.batch_size:
                    print("Drop last smaller feat batch ",len(self.batch_indices[loc][-1]))
                    self.feat_batch_indices[loc] = self.feat_batch_indices[loc][:-1]
                self.feat_batches_left[loc] = len(self.feat_batch_indices[loc])


    def extend_feat_domains(self,new_feat_domains):
        self.feat_domains += new_feat_domains
    def replace_feat_domains(self,new_feat_domains):
        if len(self.feat_domains) == 1:
            self.feat_domains += new_feat_domains
        else:
            assert len(self.feat_domains) == 3
            self.feat_domains[1] = new_feat_domains[0]
            self.feat_domains[2] = new_feat_domains[1]
    def prepare_rfc_domains(self):
        num_rfc_rounds = len(self.rfc_labels)
        self.feat_domains = [torch.arange(self.y_array.size(0))]*num_rfc_rounds

    def extend_rfc_labels(self,new_rfc_labels):
        # get rid of the place holder domains
        if len(self.feat_domains)==1:
            self.feat_domains = []
        self.rfc_labels += new_rfc_labels

    def clean_feat_domains(self):
        self.feat_domains = None
        self.feat_batch_indices = None
        self.feat_batches_left = None
        self.rfc_labels = None
    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            # mannually drop last
            if self.drop_last and len(self.batch_indices[loc][-1])<self.batch_size:
                print("Drop last smaller batch ",len(self.batch_indices[loc][-1]))
                self.batch_indices[loc] = self.batch_indices[loc][:-1]
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        img = np.load(self.root / 'images' / f'landsat_poverty_img_{self.split_idx[idx]}.npz')['x']
        if self.no_nl:
            img[-1] = 0
        img = torch.from_numpy(img).float()
        return img


    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx]

    def __len__(self):
        return len(self.targets)



class FMoW_Batched_Dataset(Dataset):
    """
    Batched dataset for FMoW. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, dataset, split, batch_size, transform, drop_last=True):
        self.split_array = dataset.split_array
        self.split_dict = dataset.split_dict
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        self.full_idxs = dataset.full_idxs[split_idx]
        self.chunk_size = dataset.chunk_size
        self.root = dataset.root

        self.metadata_array = dataset.metadata_array[split_idx]
        self.y_array = dataset.y_array[split_idx]

        self.eval = dataset.eval
        self.collate = dataset.collate
        self.metadata_fields = dataset.metadata_fields
        self.data_dir = dataset.data_dir
        self.transform = transform

        domains = dataset.metadata_array[split_idx, :2]
        self.domain_indices = [torch.nonzero((domains == loc).sum(-1) == 2).squeeze(-1)
                               for loc in domains.unique(dim=0)]
        self.domains = self.metadata_array[:, :2]
        self.num_envs = len(domains.unique(dim=0))
        print(self.num_envs)

        self.targets = self.y_array
        self.batch_size = batch_size
        self.drop_last = drop_last

        min_domain_size = np.min([len(didx) for didx in self.domain_indices])
        self.training_steps = int(min_domain_size/self.batch_size)+\
                                (not self.drop_last*(min_domain_size//self.batch_size>0))
        self.feat_domains = [torch.arange(self.y_array.size(0))] # domain 0
        self.rfc_labels = [] # domain 0
        self.train_loader_iter = None

    
    def get_feat_batch(self,feat_domain,rfc_label=False):
        batch_index = self.feat_batch_indices[feat_domain][len(self.feat_batch_indices[feat_domain]) - self.feat_batches_left[feat_domain]]
        domains = torch.zeros(batch_index.size()).to(batch_index.device)
        domains += feat_domain
        if self.train_loader_iter is not None:
            # which means align sampling:
            # inputs = self.batched_data[feat_domain][len(self.feat_batch_indices[feat_domain]) - self.feat_batches_left[feat_domain]]
            return next(self.train_loader_iter)
        else:
            inputs = torch.stack([self.transform(self.get_input(i)) for i in batch_index])
        if rfc_label:
            if len(self.rfc_labels)==len(self.feat_domains):
                targets = self.rfc_labels[feat_domain][batch_index]
            else:
                targets = self.rfc_labels[feat_domain//2][batch_index]
            return inputs, targets, domains
        else:
            return inputs, self.targets[batch_index], domains
    
    def reset_feat_batch(self,train_loader_iter=None):
        self.train_loader_iter = train_loader_iter
        self.feat_batch_indices, self.feat_batches_left = {}, {}
        if self.train_loader_iter is not None:
            assert len(self.feat_domains)==1
            self.feat_batches_left[0] = len(self.train_loader_iter)
            self.feat_batch_indices[0] = torch.arange(len(self.train_loader_iter))
        else:
            for loc, d_idx in enumerate(self.feat_domains):
                self.feat_batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
                # mannually drop last
                if self.drop_last and len(self.feat_batch_indices[loc][-1])<self.batch_size:
                    print("Drop last smaller feat batch ",len(self.batch_indices[loc][-1]))
                    self.feat_batch_indices[loc] = self.feat_batch_indices[loc][:-1]
                self.feat_batches_left[loc] = len(self.feat_batch_indices[loc])


    def extend_feat_domains(self,new_feat_domains):
        self.feat_domains += new_feat_domains
    def replace_feat_domains(self,new_feat_domains):
        if len(self.feat_domains) == 1:
            self.feat_domains += new_feat_domains
        else:
            assert len(self.feat_domains) == 3
            self.feat_domains[1] = new_feat_domains[0]
            self.feat_domains[2] = new_feat_domains[1]
    def prepare_rfc_domains(self):
        num_rfc_rounds = len(self.rfc_labels)
        self.feat_domains = [torch.arange(self.y_array.size(0))]*num_rfc_rounds

    def extend_rfc_labels(self,new_rfc_labels):
        # get rid of the place holder domains
        if len(self.feat_domains)==1:
            self.feat_domains = []
        self.rfc_labels += new_rfc_labels

    def clean_feat_domains(self):
        self.feat_domains = None
        self.feat_batch_indices = None
        self.feat_batches_left = None
        self.rfc_labels = None

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            # mannually drop last
            if self.drop_last and len(self.batch_indices[loc][-1])<self.batch_size:
                print("Drop last smaller batch ",len(self.batch_indices[loc][-1]))
                self.batch_indices[loc] = self.batch_indices[loc][:-1]
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        idx = self.full_idxs[idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx], idx

    def __len__(self):
        return len(self.targets)


class CivilComments_Batched_Dataset(Dataset):
    """
    Batched dataset for CivilComments. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, train_data, batch_size=16, drop_last=True):
        self.num_envs = 9 # civilcomments dataset has 8 attributes, plus 1 blank (no attribute)
        # print(train_data.metadata_array.size(),sum(train_data.metadata_array[:,-1]))
        meta = torch.nonzero(train_data.metadata_array[:, :8] == 1)
        indices, domains = meta[:, 0],  meta[:, 1]
        blank_indices = torch.nonzero(train_data.metadata_array[:, :8].sum(-1) == 0).squeeze()
        self.domain_indices = [blank_indices] + [indices[domains == d] for d in domains.unique()]
        domain_indices_by_group = []
        for d_idx in self.domain_indices:
            domain_indices_by_group.append(d_idx[train_data.metadata_array[d_idx][:, -2]==0])
            domain_indices_by_group.append(d_idx[train_data.metadata_array[d_idx][:, -2]==1])
        self.domain_indices = domain_indices_by_group
        
        # from wilds.common.grouper import CombinatorialGrouper
        # grouper = CombinatorialGrouper(train_data.dataset, ['y', 'black'])
        # group_array = grouper.metadata_to_group(train_data.dataset.metadata_array).numpy()
        # group_array = group_array[np.where(
        #     train_data.dataset.split_array == train_data.dataset.split_dict[
        #         'train'])]
        # print("reset domains with labels and attribute black")
        # self.domains = torch.tensor(group_array)
        # self.domain_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
        # self.domain2idx = {loc.item(): i for i, loc in enumerate(self.domains.unique())}
        # self.num_envs = len(np.unique(self.domains))
        
        
        # for loc, d_idx in enumerate(self.domain_indices):
        #     print(len(d_idx)) 

        train_data._text_array = [train_data.dataset._text_array[i] for i in train_data.indices]
        self.metadata_array = train_data.metadata_array
        self.y_array = train_data.y_array
        self.data = train_data._text_array

        self.eval = train_data.eval
        self.collate = train_data.collate
        self.metadata_fields = train_data.metadata_fields
        self.data_dir = train_data.data_dir
        self.transform = train_data.transform

        self.data = train_data._text_array
        self.targets = self.y_array
        self.domains = self.metadata_array[:, :8]
        self.batch_size = batch_size
        self.drop_last = drop_last

        min_domain_size = np.min([len(didx) for didx in self.domain_indices])
        self.training_steps = int(min_domain_size/self.batch_size)+\
                                (not self.drop_last*(min_domain_size//self.batch_size>0))
        self.feat_domains = [torch.arange(self.y_array.size(0))] # domain 0
        self.rfc_labels = [] # domain 0
        self.train_loader_iter = None

    
    def get_feat_batch(self,feat_domain,rfc_label=False):
        batch_index = self.feat_batch_indices[feat_domain][len(self.feat_batch_indices[feat_domain]) - self.feat_batches_left[feat_domain]]
        domains = torch.zeros(batch_index.size()).to(batch_index.device)
        domains += feat_domain
        if self.train_loader_iter is not None:
            # which means align sampling:
            # inputs = self.batched_data[feat_domain][len(self.feat_batch_indices[feat_domain]) - self.feat_batches_left[feat_domain]]
            return next(self.train_loader_iter)
        else:
            inputs = torch.stack([self.transform(self.get_input(i)) for i in batch_index])
        if rfc_label:
            if len(self.rfc_labels)==len(self.feat_domains):
                targets = self.rfc_labels[feat_domain][batch_index]
            else:
                targets = self.rfc_labels[feat_domain//2][batch_index]
            return inputs, targets, domains
        else:
            return inputs, self.targets[batch_index], domains
    
    def reset_feat_batch(self,train_loader_iter=None):
        self.train_loader_iter = train_loader_iter
        self.feat_batch_indices, self.feat_batches_left = {}, {}
        if self.train_loader_iter is not None:
            assert len(self.feat_domains)==1
            self.feat_batches_left[0] = len(self.train_loader_iter)
            self.feat_batch_indices[0] = torch.arange(len(self.train_loader_iter))
        else:
            for loc, d_idx in enumerate(self.feat_domains):
                self.feat_batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
                # mannually drop last
                if self.drop_last and len(self.feat_batch_indices[loc][-1])<self.batch_size:
                    print("Drop last smaller feat batch ",len(self.batch_indices[loc][-1]))
                    self.feat_batch_indices[loc] = self.feat_batch_indices[loc][:-1]
                self.feat_batches_left[loc] = len(self.feat_batch_indices[loc])


    def extend_feat_domains(self,new_feat_domains):
        self.feat_domains += new_feat_domains
    def replace_feat_domains(self,new_feat_domains):
        if len(self.feat_domains) == 1:
            self.feat_domains += new_feat_domains
        else:
            assert len(self.feat_domains) == 3
            self.feat_domains[1] = new_feat_domains[0]
            self.feat_domains[2] = new_feat_domains[1]
    def prepare_rfc_domains(self):
        num_rfc_rounds = len(self.rfc_labels)
        self.feat_domains = [torch.arange(self.y_array.size(0))]*num_rfc_rounds

    def extend_rfc_labels(self,new_rfc_labels):
        # get rid of the place holder domains
        if len(self.feat_domains)==1:
            self.feat_domains = []
        self.rfc_labels += new_rfc_labels

    def clean_feat_domains(self):
        self.feat_domains = None
        self.feat_batch_indices = None
        self.feat_batches_left = None
        self.rfc_labels = None
    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            print(len(d_idx))
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            # mannually drop last
            if self.drop_last and len(self.batch_indices[loc][-1])<self.batch_size:
                print("Drop last smaller batch ",len(self.batch_indices[loc][-1]))
                self.batch_indices[loc] = self.batch_indices[loc][:-1]
            if len(self.batch_indices[loc][-1])<=0:
                print("Drop last smaller batch ",len(self.batch_indices[loc][-1]))
                self.batch_indices[loc] = self.batch_indices[loc][:-1]
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        # print("Heyyyyyy")
        # print(self.batch_indices[domain])
        # print(self.batches_left[domain])
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        # print(domain,batch_index)
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        return self.data[idx]

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx], idx

    def __len__(self):
        return len(self.targets)


class GeneralWilds_Batched_Dataset(Dataset):
    """
    Batched dataset for Amazon, Camelyon and IwildCam. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, train_data, batch_size=16, domain_idx=0, drop_last=True):
        domains = train_data.metadata_array[:, domain_idx]
        self.domain_indices = [torch.nonzero(domains == loc).squeeze(-1) for loc in domains.unique()]
        train_data._input_array = [train_data.dataset._input_array[i] for i in train_data.indices]
        self.num_envs = len(domains.unique())

        self.metadata_array = train_data.metadata_array
        self.y_array = train_data.y_array
        self.data = train_data._input_array

        self.eval = train_data.eval
        self.collate = train_data.collate
        self.metadata_fields = train_data.metadata_fields
        self.data_dir = train_data.data_dir
        if 'iwildcam' in str(self.data_dir):
            self.data_dir = f'{self.data_dir}/train'
        self.transform = train_data.transform

        self.data = train_data._input_array
        self.targets = self.y_array
        self.domains = self.metadata_array[:, domain_idx]
        self.batch_size = batch_size
        self.drop_last = drop_last

        min_domain_size = np.min([len(didx) for didx in self.domain_indices])
        self.training_steps = int(min_domain_size/self.batch_size)+\
                                (not self.drop_last*(min_domain_size//self.batch_size>0))

        self.feat_domains = [torch.arange(self.y_array.size(0))] # domain 0
        self.rfc_labels = [] # domain 0
        self.train_loader_iter = None

    
    def get_feat_batch(self,feat_domain,rfc_label=False):
        batch_index = self.feat_batch_indices[feat_domain][len(self.feat_batch_indices[feat_domain]) - self.feat_batches_left[feat_domain]]
        domains = torch.zeros(batch_index.size()).to(batch_index.device)
        domains += feat_domain
        if self.train_loader_iter is not None:
            # which means align sampling:
            # inputs = self.batched_data[feat_domain][len(self.feat_batch_indices[feat_domain]) - self.feat_batches_left[feat_domain]]
            return next(self.train_loader_iter)
        else:
            inputs = torch.stack([self.transform(self.get_input(i)) for i in batch_index])
        if rfc_label:
            if len(self.rfc_labels)==len(self.feat_domains):
                targets = self.rfc_labels[feat_domain][batch_index]
            else:
                targets = self.rfc_labels[feat_domain//2][batch_index]
            return inputs, targets, domains
        else:
            return inputs, self.targets[batch_index], domains
    
    def reset_feat_batch(self,train_loader_iter=None):
        self.train_loader_iter = train_loader_iter
        self.feat_batch_indices, self.feat_batches_left = {}, {}
        if self.train_loader_iter is not None:
            assert len(self.feat_domains)==1
            self.feat_batches_left[0] = len(self.train_loader_iter)
            self.feat_batch_indices[0] = torch.arange(len(self.train_loader_iter))
        else:
            for loc, d_idx in enumerate(self.feat_domains):
                self.feat_batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
                # mannually drop last
                if self.drop_last and len(self.feat_batch_indices[loc][-1])<self.batch_size:
                    print("Drop last smaller feat batch ",len(self.batch_indices[loc][-1]))
                    self.feat_batch_indices[loc] = self.feat_batch_indices[loc][:-1]
                self.feat_batches_left[loc] = len(self.feat_batch_indices[loc])

    def extend_feat_domains(self,new_feat_domains):
        self.feat_domains += new_feat_domains
    def replace_feat_domains(self,new_feat_domains):
        if len(self.feat_domains) == 1:
            self.feat_domains += new_feat_domains
        else:
            assert len(self.feat_domains) == 3
            self.feat_domains[1] = new_feat_domains[0]
            self.feat_domains[2] = new_feat_domains[1]
    def prepare_rfc_domains(self):
        num_rfc_rounds = len(self.rfc_labels)
        self.feat_domains = [torch.arange(self.y_array.size(0))]*num_rfc_rounds

    def extend_rfc_labels(self,new_rfc_labels):
        # get rid of the place holder domains
        if len(self.feat_domains)==1:
            self.feat_domains = []
        self.rfc_labels += new_rfc_labels

    def clean_feat_domains(self):
        self.feat_domains = None
        self.feat_batch_indices = None
        self.feat_batches_left = None
        self.rfc_labels = None

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            # mannually drop last
            if self.drop_last and len(self.batch_indices[loc][-1])<self.batch_size:
                print("Drop last smaller batch ",len(self.batch_indices[loc][-1]))
                self.batch_indices[loc] = self.batch_indices[loc][:-1]
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        if isinstance(self.data_dir, str) and 'amazon' in self.data_dir:
            return self.data[idx]
        else:
            # All images are in the train folder
            img_path = f'{self.data_dir}/{self.data[idx]}'
            img = Image.open(img_path)
            if isinstance(self.data_dir, str) and not ('iwildcam' in self.data_dir):
                img = img.convert('RGB')
            return img

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx], idx

    def __len__(self):
        return len(self.targets)
