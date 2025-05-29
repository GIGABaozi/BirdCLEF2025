import torch
import torch.nn as nn
import timm

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            num_classes=0
        )
        
        # 获取特征维度
        n_features = self.model.num_features
        
        # 添加分类头
        self.head = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.num_classes)
        )
        
    def forward(self, x):
        # 输入形状: (batch_size, channels, height, width)
        features = self.model(x)
        # 输出形状: (batch_size, num_classes)
        return self.head(features) 