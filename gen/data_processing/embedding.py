from functools import reduce
from dm.data_processing.internal_coord import ZmatConversion
import torch
import numpy as np

def composite_function(*func):
      
    def compose(f, g):
        return lambda x : f(g(x))
              
    return reduce(compose, func, lambda x : x)

class Embedding():
    def __init__(self,transform_types=None,data_dir=None,**kwargs):
        self.data_dir = data_dir
        self.encoder_list = []
        self.decoder_list = []
        if transform_types is None:
            self.transforms = None
            self.encoder = lambda x: x
            self.decoder = lambda x: x
        else:
            self.transforms = self.parse_transforms(transform_types)
            self.encoder = self.get_transform(encode=True)
            self.decoder = self.get_transform(encode=False)
 
    def get_transform(self,encode=True):
        for transform in self.transforms:
            if encode:
                self.encoder_list.append(transform.encode)
            else:
                self.decoder_list.append(transform.decode)
        if encode:
            return composite_function(*self.encoder_list)
        else:
            return composite_function(*self.decoder_list)

    def parse_transforms(self,transform_types):
        transforms = []
        for transform_type in transform_types:
            if transform_type == "zmat":
                transforms.append(ZmatConversion())
            elif transform_type == "normalize":
                data = torch.tensor(np.load(self.data_dir))
                mean = 0.0
                std = data.std()
                transforms.append(Normalize(mean,std))
            else:
                raise NotImplementedError
        return transforms

class Normalize():
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
        print(self.std)

    def encode(self,x):
        return (x-self.mean)/self.std

    def decode(self,x):
        return x*self.std+self.mean