from random import shuffle
import torch.distributed as dist
import torch
from collections import OrderedDict

class MultiData_DistributedSampler(object):
    def __init__(self, name2dataset,  shuffle, seed: int = 0, drop_last: bool = False):
        #name2dataset (ordered dict) (the order is the same as MixedDataset)
        self.name2sampler = OrderedDict()
        self.name2dataset = name2dataset
        self.num_replicas = dist.get_world_size()
        self.shuffle = shuffle
        self.seed = seed
        for name, dataset in name2dataset.items():
            self.name2sampler[name] = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)
            #print(f'{name} dataset #={len(dataset)}  sampler #={len(self.name2sampler[name])}')
        #print(f'MultiData #={len(self)}')
        #debug
        # import pickle
        # self.set_epoch(0)
        # indices = list(self.__iter__())
        # with open(f'debug/sampler/epoch_0_indices_rank{dist.get_rank()}.pkl','wb') as f:
        #     pickle.dump(indices, f)

        # self.set_epoch(1)
        # indices = list(self.__iter__())
        # with open(f'debug/sampler/epoch_1_indices_rank{dist.get_rank()}.pkl','wb') as f:
        #     pickle.dump(indices, f)
        # print('save')

        
        


    def __iter__(self):
        indices, offset = [], 0
        for name, sampler in self.name2sampler.items():
            indices.extend([i+offset for i in sampler])
            offset += len(self.name2dataset[name])   
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 1)
            shuffle_indices = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffle_indices]
        return iter(indices) #no need to do subsampler anymore

    def __len__(self):
        return sum([len(sampler) for name, sampler in self.name2sampler.items()])
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        for name, sampler in self.name2sampler.items():
            sampler.set_epoch(epoch)

        

