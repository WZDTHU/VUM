import json
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .c2i_data import ImagenetImageDataset
from .utils.sampler import get_sampler




class TrainDataLoader():
    def __init__(self, data_config):
        super().__init__()

        self.batch_size = data_config.dataloader.batch_size
        self.num_workers = data_config.dataloader.num_workers
        self.data_type = data_config.data_type
    
        if self.data_type == 'c2i_image':
            self.train_dataset = ImagenetImageDataset(**OmegaConf.to_container(data_config.dataset))
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.train_dataset)

    def train_dataloader(self, rank, world_size, global_batch_size, max_steps, resume_steps, seed):
        collate_fn = None

        sampler = get_sampler(
            self.train_dataset, rank, world_size, global_batch_size, max_steps, resume_steps, seed
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
        )
        
        
        