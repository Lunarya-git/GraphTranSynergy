"""
run_one_epoch.py
----------------
Runs training_GraphTransformer.py for exactly ONE epoch across all 5 folds.
All other logic (data loading, evaluation, saving) is identical to the original.

Usage (from project root, inside the venv):
    python run_one_epoch.py
"""

import random
import time
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import DataLoader
from model.GraphTransformer import GraphTransformerNet
from utils_test import TestbedDataset, save_AUCs
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, cohen_kappa_score,
    accuracy_score, precision_score, recall_score, balanced_accuracy_score
)
from sklearn import metrics

# ── Hyperparameters ──────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE  = 256
LR               = 0.0005
LOG_INTERVAL     = 20
NUM_EPOCHS       = 1          # <<< ONE EPOCH ONLY
NUM_FOLDS        = 5          # set to 1 to run a single fold quickly

print(f"Learning rate : {LR}")
print(f"Epochs        : {NUM_EPOCHS}  (hard-coded to 1 in run_one_epoch.py)")
print(f"Folds         : {NUM_FOLDS}")

datafile = "new_labels_0_10"

# ── Device ───────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device        : {device}")

# ── Loss ─────────────────────────────────────────────────────────────────────
loss_fn = nn.CrossEntropyLoss()


# ── Train / Predict helpers ──────────────────────────────────────────────────
def train(model, device, drug1_loader, drug2_loader, optimizer, epoch):
    print(f"Training on {len(drug1_loader.dataset)} samples...")
    model.train()
    for batch_idx, data in enumerate(zip(drug1_loader, drug2_loader)):
        data1, data2 = data[0].to(device), data[1].to(device)
        y = data[0].y.view(-1, 1).long().to(device).squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(
                f"  Train epoch {epoch} "
                f"[{batch_idx * len(data1.x)}/{len(drug1_loader.dataset)} "
                f"({100. * batch_idx / len(drug1_loader):.0f}%)]"
                f"  Loss: {loss.item():.6f}"
            )


def predicting(model, device, drug1_loader, drug2_loader):
    model.eval()
    total_preds      = torch.Tensor()
    total_labels     = torch.Tensor()
    total_prelabels  = torch.Tensor()
    print(f"Predicting on {len(drug1_loader.dataset)} samples...")
    with torch.no_grad():
        for data in zip(drug1_loader, drug2_loader):
            data1, data2 = data[0].to(device), data[1].to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).cpu().data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds     = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels    = torch.cat((total_labels, data[0].y.view(-1, 1).cpu()), 0)
    return (
        total_labels.numpy().flatten(),
        total_preds.numpy().flatten(),
        total_prelabels.numpy().flatten(),
    )


# ── Data ─────────────────────────────────────────────────────────────────────
drug1_data = TestbedDataset(root="data", dataset=datafile + "_drug1")
drug2_data = TestbedDataset(root="data", dataset=datafile + "_drug2")

length = len(drug1_data)
pot    = int(length / 5)
print(f"Dataset size  : {length}  |  fold size: {pot}")

random_num = random.sample(range(0, length), length)

# ── Cross-validation loop ─────────────────────────────────────────────────────
run_start = time.time()
for i in range(NUM_FOLDS):
    print(f"\n{'='*60}")
    print(f"  FOLD {i+1} / {NUM_FOLDS}")
    print(f"{'='*60}")
    fold_start = time.time()

    test_num  = random_num[pot * i : pot * (i + 1)]
    train_num = random_num[: pot * i] + random_num[pot * (i + 1) :]

    drug1_loader_train = DataLoader(drug1_data[train_num], batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug1_loader_test  = DataLoader(drug1_data[test_num],  batch_size=TEST_BATCH_SIZE,  shuffle=None)
    drug2_loader_train = DataLoader(drug2_data[train_num], batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_test  = DataLoader(drug2_data[test_num],  batch_size=TEST_BATCH_SIZE,  shuffle=None)

    model     = GraphTransformerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Output paths
    out_dir = "data/result/sun_pre_test/GraphTransformer"
    os.makedirs(out_dir, exist_ok=True)
    model_file  = f"{out_dir}/GraphTransformerNet(DrugA_DrugB){i}--model_{datafile}.model"
    result_file = f"{out_dir}/GraphTransformerNet(DrugA_DrugB){i}--result_{datafile}.csv"
    auc_file    = f"{out_dir}/GraphTransformerNet(DrugA_DrugB){i}--AUCs--{datafile}.txt"

    with open(auc_file, "w") as f:
        f.write("Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL\n")

    # Load checkpoint if one exists
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, weights_only=True))
        print(f"  Loaded existing checkpoint: {model_file}")

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
        epoch_elapsed = time.time() - epoch_start

        AUC        = roc_auc_score(T, S)
        precision, recall_curve, _ = metrics.precision_recall_curve(T, S)
        PR_AUC     = metrics.auc(recall_curve, precision)
        BACC       = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR        = tp / (tp + fn)
        PREC       = precision_score(T, Y)
        ACC        = accuracy_score(T, Y)
        KAPPA      = cohen_kappa_score(T, Y)
        RECALL     = recall_score(T, Y)

        AUCs = [epoch + 1, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL]
        print(f"\n  Epoch {epoch+1} results — AUC: {AUC:.4f} | PR-AUC: {PR_AUC:.4f} | ACC: {ACC:.4f} | Time: {epoch_elapsed:.1f}s")

        if best_auc < AUC:
            best_auc = AUC
            print(f"  *** New best AUC: {best_auc:.4f} — saving model ***")
            save_AUCs(AUCs, auc_file)
            torch.save(model.state_dict(), model_file)
            txtDF = pd.DataFrame([test_num, list(T), list(Y), list(S)])
            txtDF.to_csv(result_file, index=False, header=False)

    fold_elapsed = time.time() - fold_start
    print(f"\n  Fold {i+1} complete in {fold_elapsed:.1f}s")

total_elapsed = time.time() - run_start
print(f"\nDone. Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
print(f"Results saved to: data/result/sun_pre_test/GraphTransformer/")
