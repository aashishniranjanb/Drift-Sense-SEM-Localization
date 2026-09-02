import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, precision_recall_curve, auc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class PatchDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        ref = torch.tensor(row['ref'], dtype=torch.float32).unsqueeze(0) / 255.0
        search = torch.tensor(row['search'], dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(row['label'], dtype=torch.float32)
        return ref, search, label

class SiameseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x1, x2):
        f1 = self.features(x1).view(x1.size(0), -1)
        f2 = self.features(x2).view(x2.size(0), -1)
        dist = torch.abs(f1 - f2)
        return self.fc(dist).squeeze(1)

class TwoStreamCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(nn.Linear(64*3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x1, x2):
        f1 = self.features(x1).view(x1.size(0), -1)
        f2 = self.features(x2).view(x2.size(0), -1)
        combined = torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)
        return self.fc(combined).squeeze(1)

def extract_patch(img, cx, cy, size=64):
    h, w = img.shape
    x1 = max(0, int(cx - size//2))
    y1 = max(0, int(cy - size//2))
    x2 = min(w, int(cx + size//2))
    y2 = min(h, int(cy + size//2))
    patch = img[y1:y2, x1:x2]
    if patch.shape != (size, size):
        patch = cv2.resize(patch, (size, size))
    return patch

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'phase2_dev'))
    df_pairs = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    
    print("Generating patches...")
    all_patches = []
    
    for idx, row in df_pairs.iterrows():
        ref_path = os.path.join(data_dir, row["reference_path"])
        search_path = os.path.join(data_dir, row["search_path"])
        
        ref_img = cv2.imread(ref_path, 0)
        search_img = cv2.imread(search_path, 0)
        if ref_img is None or search_img is None: continue
        
        scale = float(row.get('gt_scale', 10.0))
        if scale < 0.1: scale = 10.0
        theta = float(row.get("gt_theta", 0.0))
        gt_found = int(row.get("gt_found", 1))
        gt_x = float(row.get("gt_x", search_img.shape[1]/2))
        gt_y = float(row.get("gt_y", search_img.shape[0]/2))
        
        # Resize/rotate ref to match search domain
        ref_w = int(round(ref_img.shape[1] / scale))
        ref_h = int(round(ref_img.shape[0] / scale))
        if ref_w < 10 or ref_h < 10: continue
        ref_resized = cv2.resize(ref_img, (ref_w, ref_h))
        
        if abs(theta) > 0.1:
            M = cv2.getRotationMatrix2D((ref_w/2, ref_h/2), theta, 1.0)
            ref_template = cv2.warpAffine(ref_resized, M, (ref_w, ref_h))
        else:
            ref_template = ref_resized
            
        corr = cv2.matchTemplate(search_img, ref_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
        
        ref_patch = cv2.resize(ref_template, (64, 64))
        
        # Extract candidates
        # Find local maxima in corr for hard negatives
        c_work = corr.copy()
        
        candidates = []
        for _ in range(3):
            _, val, _, loc = cv2.minMaxLoc(c_work)
            cx = loc[0] + ref_w/2
            cy = loc[1] + ref_h/2
            candidates.append((cx, cy, val))
            # suppress
            y1 = max(0, loc[1]-20)
            y2 = min(c_work.shape[0], loc[1]+20)
            x1 = max(0, loc[0]-20)
            x2 = min(c_work.shape[1], loc[0]+20)
            c_work[y1:y2, x1:x2] = -1
            
        for cx, cy, score in candidates:
            cand_patch = extract_patch(search_img, cx, cy, size=64)
            if gt_found == 1:
                err = np.hypot(cx - gt_x, cy - gt_y)
                label = 1 if err <= 10.0 else 0
            else:
                label = 0
                
            all_patches.append({
                'pair_id': row['pair_id'],
                'ref': ref_patch,
                'search': cand_patch,
                'label': label,
                'corr_score': score
            })

    df_p = pd.DataFrame(all_patches)
    print(f"Total patches: {len(df_p)}")
    
    # Stratified Split by pair_id to prevent leakage
    pairs = df_p['pair_id'].unique()
    np.random.seed(42)
    np.random.shuffle(pairs)
    train_pairs = pairs[:int(len(pairs)*0.6)]
    val_pairs = pairs[int(len(pairs)*0.6):int(len(pairs)*0.8)]
    test_pairs = pairs[int(len(pairs)*0.8):]
    
    train_data = df_p[df_p['pair_id'].isin(train_pairs)].to_dict('records')
    val_data = df_p[df_p['pair_id'].isin(val_pairs)].to_dict('records')
    test_data = df_p[df_p['pair_id'].isin(test_pairs)].to_dict('records')
    
    y_test_true = [x['label'] for x in test_data]
    y_test_ncc = [x['corr_score'] for x in test_data]
    
    # Avoid zero_division warnings by checking if any positive samples
    if sum(y_test_true) == 0:
        f1_a = 0
        auc_a = 0
    else:
        f1_a = f1_score(y_test_true, (np.array(y_test_ncc) > 0.8).astype(int), zero_division=0)
        auc_a = roc_auc_score(y_test_true, y_test_ncc)
    
    def train_model(model, train_loader, epochs=10):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for ep in range(epochs):
            model.train()
            for ref, search, label in train_loader:
                optimizer.zero_grad()
                out = model(ref, search)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
        return model

    train_loader = DataLoader(PatchDataset(train_data), batch_size=32, shuffle=True)
    val_loader = DataLoader(PatchDataset(val_data), batch_size=32)
    test_loader = DataLoader(PatchDataset(test_data), batch_size=32)
    
    torch.manual_seed(42)
    model_b = train_model(SiameseCNN(), train_loader)
    torch.manual_seed(42)
    model_c = train_model(TwoStreamCNN(), train_loader)
    
    def eval_model(model, loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ref, search, label in loader:
                out = model(ref, search)
                preds.extend(out.numpy())
                trues.extend(label.numpy())
        return trues, preds
        
    _, preds_b = eval_model(model_b, test_loader)
    _, preds_c = eval_model(model_c, test_loader)
    
    if sum(y_test_true) == 0:
        auc_b, auc_c = 0, 0
    else:
        auc_b = roc_auc_score(y_test_true, preds_b)
        auc_c = roc_auc_score(y_test_true, preds_c)
        
    f1_b = f1_score(y_test_true, (np.array(preds_b) > 0.5).astype(int), zero_division=0)
    f1_c = f1_score(y_test_true, (np.array(preds_c) > 0.5).astype(int), zero_division=0)
    
    res = "# Phase V20.2 Patch Verifier Baseline Results\n\n"
    res += f"- **V20.2-A (Classical NCC)**: AUC = {auc_a:.4f}, F1 = {f1_a:.4f}\n"
    res += f"- **V20.2-B (Siamese CNN)**: AUC = {auc_b:.4f}, F1 = {f1_b:.4f}\n"
    res += f"- **V20.2-C (Two-Stream CNN)**: AUC = {auc_c:.4f}, F1 = {f1_c:.4f}\n\n"
    res += "Models trained over 10 epochs using physical hard-negative patches.\n"
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')), exist_ok=True)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'V20_2_ABC_RESULTS.md')), 'w') as f:
        f.write(res)
        
    print(res)

if __name__ == "__main__":
    main()

