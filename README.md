# BirdCLEF 2025 鸟类声音识别

这个项目是用于BirdCLEF 2025比赛的鸟类声音识别系统。该系统使用深度学习方法来识别鸟类的声音。

## 项目结构

```
.
├── src/
│   ├── config/
│   │   └── config.py          # 配置文件
│   ├── data/
│   │   ├── audio_processing.py # 音频处理函数
│   │   └── dataset.py         # 数据集类
│   ├── models/
│   │   └── model.py           # 模型定义
│   ├── training/
│   │   └── trainer.py         # 训练器类
│   ├── utils/
│   │   └── utils.py           # 工具函数
│   └── train.py               # 主训练脚本
├── requirements.txt           # 项目依赖
└── README.md                 # 项目说明
```

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd BirdCLEF2025
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：
   - 将训练数据放在 `kaggle/input/birdclef-2025/train_audio` 目录下
   - 将训练标签CSV文件放在 `kaggle/input/birdclef-2025/train.csv`

2. 训练模型：
```bash
python src/train.py
```

## 模型架构

- 使用EfficientNet-B0作为基础模型
- 添加自定义分类头
- 使用BCE损失函数
- 使用AdamW优化器和余弦退火学习率调度

## 数据预处理

- 音频转换为梅尔频谱图
- 标准化处理
- 数据增强（待实现）

## 训练策略

- 使用分层K折交叉验证
- 使用早停策略
- 保存最佳模型权重

## 评估指标

- 使用每个类别的AUC分数
- 计算平均AUC作为主要评估指标 