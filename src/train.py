import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from src.config.config import CFG
from src.data.dataset import BirdCLEFDataset
from src.data.audio_processing import generate_spectrograms
from src.models.model import BirdCLEFModel
from src.training.trainer import Trainer
from src.utils.utils import set_seed

def main():
    # 设置配置
    cfg = CFG()
    set_seed(cfg.seed)
    
    # 创建输出目录
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    train_df = pd.read_csv(cfg.train_csv)
    
    # 在debug模式下只使用少量样本
    if cfg.debug:
        print("Debug mode: Using only 10 samples")
        train_df = train_df.head(10)
    
    # 确保taxonomy.csv存在并加载类别数量
    if not os.path.exists(cfg.taxonomy_csv):
        raise FileNotFoundError(f"taxonomy.csv not found at {cfg.taxonomy_csv}")
    
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    cfg.num_classes = len(taxonomy_df['primary_label'].unique())
    print(f"Number of classes: {cfg.num_classes}")
    
    # 生成声谱图
    if cfg.LOAD_DATA:
        spectrograms = generate_spectrograms(train_df, cfg)
    else:
        spectrograms = None
    
    # 创建交叉验证折
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    
    # 训练每个折
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue
            
        print(f"\nTraining Fold {fold}")
        
        # 创建数据集
        train_dataset = BirdCLEFDataset(
            train_df.iloc[train_idx],
            cfg,
            spectrograms=spectrograms,
            mode="train"
        )
        
        val_dataset = BirdCLEFDataset(
            train_df.iloc[val_idx],
            cfg,
            spectrograms=spectrograms,
            mode="val"
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        
        # 创建模型
        model = BirdCLEFModel(cfg)
        
        # 创建训练器
        trainer = Trainer(model, cfg)
        
        # 训练模型
        trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main() 