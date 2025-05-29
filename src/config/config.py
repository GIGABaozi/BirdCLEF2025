import torch

class CFG:
    seed = 42
    debug = False  
    apex = False
    print_freq = 100
    num_workers = 2

    PROJECT_ROOT = "."

    OUTPUT_DIR = f"{PROJECT_ROOT}/kaggle/working/"
    
    train_datadir = f"{PROJECT_ROOT}/kaggle/input/birdclef-2025/train_audio"
    train_csv = f"{PROJECT_ROOT}/kaggle/input/birdclef-2025/train.csv"
    test_soundscapes = f"{PROJECT_ROOT}/kaggle/input/birdclef-2025/test_soundscapes"
    submission_csv = f"{PROJECT_ROOT}/kaggle/input/birdclef-2025/sample_submission.csv"
    taxonomy_csv = f"{PROJECT_ROOT}/kaggle/input/birdclef-2025/taxonomy.csv"
    cache_dir = f"{PROJECT_ROOT}/kaggle/input/efficientnet-b0"
    
    spectrogram_npy = f"{PROJECT_ROOT}/kaggle/input/birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy"
 
    model_name = 'efficientnet_b0'  
    pretrained = True
    in_channels = 1

    LOAD_DATA = True  
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)
    
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10  
    batch_size = 32  
    criterion = 'BCEWithLogitsLoss'

    n_fold = 4
    selected_folds = [0, 1, 2, 3, 4]   

    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5
  
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5  
    mixup_alpha = 0.5  
    
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0] 