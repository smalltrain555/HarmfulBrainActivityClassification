import os
import gc
import random
import shutil
from time import time
import typing as tp
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp

from generate_data import create_train_data

class CFG:
    in_h = 800
    in_w = 800
    in_channels = 1
    max_epoch = 30
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience =  10
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"
    gsn_prob = 0.5
    below_10 = 0
    above_10 = 1

    model_name = 'efficientnet_b2'
    pretrained_model = './pretrained/timm_efficientnet_b2.ra_in1k/pytorch_model.bin'    
    model_save_name = f'{model_name}_{in_channels}x{in_h}x{in_w}_gsn{gsn_prob}_below10_{below_10}_above10_{above_10}'


RANDAM_SEED = CFG.seed
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)


## Read data
train = create_train_data('./hms')
TRAIN_SPEC_SPLIT = './hms/train_data'

if not os.path.exists(TRAIN_SPEC_SPLIT):
    os.makedirs(TRAIN_SPEC_SPLIT)

## split train data and valid data
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDAM_SEED)
train["fold"] = -1
for fold_id, (_, val_idx) in enumerate(sgkf.split(train, y=train["expert_consensus"], groups=train["patient_id"])):
    train.loc[val_idx, "fold"] = fold_id

train = train.groupby("new_id").head(1).reset_index(drop=True)
print(train.shape)

## define model
class HMSHBACSpecModel(nn.Module):
    def __init__(self,
                 model_name: str,
                 pretrained: bool,
                 in_channels: int,
                 num_classes: int,):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained,
            num_classes=num_classes, in_chans=in_channels,
            pretrained_cfg_overlay=dict(file=CFG.pretrained_model))

    def forward(self, x):
        h = self.model(x)
        return h


## define dataset
FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]

class HMSHBACSpecDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths: tp.Sequence[FilePath],
                 labels: tp.Sequence[Label],
                 weights:tp.Sequence[Label],
                 transform: A.Compose,
                 in_chl: int=1,
                 pad_method: int=0):
        self.image_paths = image_paths
        self.labels = labels
        self.weights = weights
        self.transform = transform
        self.in_chl = in_chl
        self.pad_method = pad_method

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]
        weight = self.weights[index]

        img = np.load(img_path)
        img_50s = img['image_50s']
        img_10s = img['image_10s']
        img_10m = img['image_10m']

        _img = np.zeros((800, 800), dtype=np.float32)
        for i in range(4):
            _img[i*200:(i+1)*200,:500] = img_50s[:200,:500, i]
            _img[i*100:(i+1)*100,500:500+291] = img_10s[:, :, i]
            _img[400+i*100:400+(i+1)*100,500:] = img_10m[i*100:(i+1)*100, :]

        _img = self._apply_transform(_img)

        return {"data": _img, "target": label, "weight": weight}


    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img


def get_transforms(CFG):
    train_transform = A.Compose([
        A.GaussNoise((10,50), mean=0, p=CFG.gsn_prob),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)])
    val_transform = A.Compose([
        ToTensorV2(p=1.0)])
    return train_transform, val_transform


## loss
class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y,  dim=1)
        loss = super().forward(y, t)
        return loss


class KLDivLossWithLogitsForVal(nn.KLDivLoss):
    def __init__(self):
        """"""
        super().__init__(reduction="batchmean")
        self.log_prob_list  = []
        self.label_list = []

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        self.log_prob_list.append(y.numpy())
        self.label_list.append(t.numpy())
        
    def compute(self):
        log_prob = np.concatenate(self.log_prob_list, axis=0)
        label = np.concatenate(self.label_list, axis=0)
        final_metric = super().forward(torch.from_numpy(log_prob),
                                       torch.from_numpy(label)).item()
        self.log_prob_list = []
        self.label_list = []
        
        return final_metric
    

## train config
def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic  # type: ignore


def to_device(tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
              device: torch.device, *args, **kwargs):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_path_label(val_fold, train_all: pd.DataFrame):
    # Get file path and target info.
    train_idx = train_all[train_all["fold"] != val_fold].index.values
    val_idx   = train_all[train_all["fold"] == val_fold].index.values
    img_paths = []
    eeg_paths = []
    labels = train_all[CLASSES].values

    for train_info in train_all.groupby("new_id"):
        eeg_id = train_info[1]['eeg_id'].values[0]
        votes = train_info[1]['sum_votes'].values[0]
        max_vote_prob = train_info[1]['max_vote_prob'].values[0]
        seconds_min = int(train_info[1]['spectrogram_label_offset_seconds_min'].values[0])
        start_second = int(train_info[1]['eeg_label_offset_seconds_min'].values[0])
        eeg_new_key = f'{eeg_id}_{seconds_min}_{start_second}'
        img_path = os.path.join(TRAIN_SPEC_SPLIT, f"{eeg_new_key}.npz")
        img_paths.append(img_path)

    train_data = {
        "image_paths": [img_paths[idx] for idx in train_idx],
        "labels": [labels[idx].astype("float32") for idx in train_idx],
        "weights": [weights[idx] for idx in train_idx]
    }

    val_data = {
        "image_paths": [img_paths[idx] for idx in val_idx],
        "labels": [labels[idx].astype("float32") for idx in val_idx],
        "weights": [weights[idx] for idx in train_idx]
    }
    
    return train_data, val_data, train_idx, val_idx


def train_one_fold(CFG, val_fold, train_all, output_path):
    # Main
    torch.backends.cudnn.benchmark = True
    set_random_seed(CFG.seed, deterministic=CFG.deterministic)
    device = torch.device(CFG.device)
    
    train_path_label, val_path_label, _, _ = get_path_label(val_fold, train_all)
    train_transform, val_transform = get_transforms(CFG)
    
    train_dataset = HMSHBACSpecDataset(**train_path_label, transform=train_transform, 
                                    in_chl=CFG.in_channels, pad_method=CFG.pad_method)
    val_dataset = HMSHBACSpecDataset(**val_path_label, transform=val_transform, 
                                    in_chl=CFG.in_channels, pad_method=CFG.pad_method)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size, 
                                               num_workers=4, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CFG.batch_size, 
                                             num_workers=4, shuffle=False, drop_last=False)
    
    model = HMSHBACSpecModel(model_name=CFG.model_name, 
                             pretrained=True, 
                             num_classes=6, 
                             in_channels=CFG.in_channels)
    model.to(device)
    
    optimizer = optim.AdamW(params=model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, epochs=CFG.max_epoch,
                                        pct_start=0.0, steps_per_epoch=len(train_loader),
                                        max_lr=CFG.lr, div_factor=25, final_div_factor=4.0e-01)
    
    loss_func = KLDivLossWithLogits().to(device)
    loss_func_val = KLDivLossWithLogitsForVal()
    
    use_amp = CFG.enable_amp
    scaler = amp.GradScaler(enabled=use_amp)
    
    best_val_loss = 1.0e+09
    best_epoch = 0
    train_loss = 0
    
    for epoch in range(1, CFG.max_epoch + 1):
        epoch_start = time()
        model.train()
        for batch in train_loader:
            batch = to_device(batch, device)
            x, t = batch["data"], batch["target"]

            with amp.autocast(use_amp):
                y = model(x)
                loss = loss_func(y, t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
            
        model.eval()
        for batch in val_loader:
            x, t = batch["data"], batch["target"]
            x = to_device(x, device)
            with torch.no_grad(), amp.autocast(use_amp):
                y = model(x)
            y = y.detach().cpu().to(torch.float32)
            loss_func_val(y, t)
        val_loss = loss_func_val.compute()        
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss

            for filename in os.listdir(output_path):
                file_path = os.path.join(output_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            torch.save(model.state_dict(), os.path.join(output_path, f'snapshot_epoch_{epoch}_val_loss_{best_val_loss:.3f}.pth'))
            shutil.copy(os.path.join(output_path, f'snapshot_epoch_{epoch}_val_loss_{best_val_loss:.3f}.pth'),
                        os.path.join(output_path.replace(f'/fold{fold_id}', '/best_models'), f'snapshot_{val_fold}_best_model.pth'))
        
        elapsed_time = time() - epoch_start
        print(f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, elapsed_time: {elapsed_time: .3f}")
        
        if epoch - best_epoch > CFG.es_patience:
            print("Early Stopping!")
            break
            
        train_loss = 0
            
    return val_fold, best_epoch, best_val_loss


## training
score_list = []
for fold_id in FOLDS:
    output_path = os.path.join('work_dirs', CFG.model_save_name, f'fold{fold_id}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(output_path.replace(f'/fold{fold_id}', '/best_models')):
        os.makedirs(output_path.replace(f'/fold{fold_id}', '/best_models'))

    print(f"[fold{fold_id}]")
    score_list.append(train_one_fold(CFG, fold_id, train, output_path))


## inference
print('score_list: ', score_list)

def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())
        
    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr

label_arr = train[CLASSES].values
oof_pred_arr = np.zeros((len(train), N_CLASSES))
score_list = []

for fold_id in range(N_FOLDS):
    print(f"\n[fold {fold_id}]")
    device = torch.device(CFG.device)

    # # get_dataloader
    _, val_path_label, _, val_idx = get_path_label(fold_id, train)
    _, val_transform = get_transforms(CFG)
    val_dataset = HMSHBACSpecDataset(**val_path_label, transform=val_transform, 
                                     in_chl=CFG.in_channels, pad_method=CFG.pad_method)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CFG.batch_size, 
                                             num_workers=4, shuffle=False, drop_last=False)
    
    # # get model
    model_path = os.path.join('./work_dirs', CFG.model_save_name, 'best_models', f"snapshot_{fold_id}_best_model.pth")
    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=CFG.in_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # # inference
    val_pred = run_inference_loop(model, val_loader, device)
    oof_pred_arr[val_idx] = val_pred
    
    del val_idx, val_path_label
    del model, val_loader
    torch.cuda.empty_cache()
    gc.collect()


## OOF score
from utils.kaggle_kl_div import score
true = train[["new_id"] + CLASSES].copy()
oof = pd.DataFrame(oof_pred_arr, columns=CLASSES)
oof.insert(0, "new_id", train["new_id"])
cv_score = score(solution=true, submission=oof, row_id_column_name='new_id')
