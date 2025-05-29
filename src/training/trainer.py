import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

class Trainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        
        # 设置损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        
        # 设置学习率调度器
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            self.optimizer.zero_grad()
            
            mel_spec = batch['mel_spec'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(mel_spec)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
            
        self.scheduler.step()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                mel_spec = batch['mel_spec'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(mel_spec)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_preds.append(outputs.sigmoid().cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # 计算每个类别的AUC
        auc_scores = []
        for i in range(all_labels.shape[1]):
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                auc_scores.append(auc)
            except:
                pass
        
        mean_auc = np.mean(auc_scores)
        return total_loss / len(val_loader), mean_auc
    
    def train(self, train_loader, val_loader):
        best_auc = 0
        
        for epoch in range(self.cfg.epochs):
            print(f"\nEpoch {epoch + 1}/{self.cfg.epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), f"{self.cfg.OUTPUT_DIR}/best_model.pth")
                print(f"Saved best model with AUC: {best_auc:.4f}") 