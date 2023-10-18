from torch.utils.data import DataLoader
import copy

from dm.data_processing.embedding import Embedding
from dm.data_processing.dataset import TrajData, LiveSimulation
from dm.simulations._setup_distributions import setup_distribution

class DataHandler():
    def __init__(self, cfg, mode="train", transform_types=["normalize"]):
        self.cfg = cfg
        self.device = cfg.device
        self.distribution = setup_distribution(cfg, cfg.dataset.potential,cfg.dataset)
        if cfg.dataset.testing_data is not None:
            self.embedding = Embedding(transform_types,data_dir=cfg.dataset.testing_data,device=self.device)
            self.encoder = self.embedding.encoder
            self.decoder = self.embedding.decoder
            self.test_dataloader = self.setup_dataloader(cfg.dataset.testing_data)
        if cfg.dataset.training_data is not None:        
            if mode != "sample": #set up dataloader for training
                self.dataloader = self.setup_dataloader(cfg.dataset.training_data)
        else:
            assert hasattr(self,"sampler")   
            self.dataloader = LiveSimulation(self.sampler,batch_size=cfg.train_parameters.batch_size,device=cfg.device)
            self.encoder = lambda x: x
            self.decoder = lambda x: x
            self.test_dataloader = self.dataloader

    def setup_dataloader(self,dir):
        dataset = TrajData(dir, encoder=self.encoder,device=self.device)
        self.dataloader = DataLoader(copy.deepcopy(dataset), batch_size=self.cfg.train_parameters.batch_size,shuffle=True, num_workers=2)
        return self.dataloader