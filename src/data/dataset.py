import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class BirdCLEFDataset(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms
        
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        
        self.label_encoder = {species: idx for idx, species in enumerate(self.species_ids)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.spectrograms is not None:
            mel_spec = self.spectrograms[row['filename']]
        else:
            filepath = os.path.join(self.cfg.train_datadir, row['filename'])
            mel_spec = np.load(filepath)
            
        mel_spec = torch.from_numpy(mel_spec).float()
        
        if self.mode == "train":
            # 在这里添加数据增强
            pass
            
        # 创建标签
        label = np.zeros(self.num_classes)
        for species in row['primary_label'].split():
            if species in self.label_encoder:
                label[self.label_encoder[species]] = 1
                
        return {
            'mel_spec': mel_spec,
            'label': torch.from_numpy(label).float(),
            'filename': row['filename']
        } 