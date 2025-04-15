import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, AUROC
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os

def Fmax(dataloader, model):
    model.eval()
    torch.cuda.empty_cache()
    preds, targets = [], []
    for X, target in dataloader:
        pred = model(X)
        preds.append(pred)
        targets.append(target)

    pred = torch.cat(preds)
    target = torch.cat(targets)
    return f1_max(pred, target)

def f1_max(pred, target):
    """
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


class GoAnnots(Dataset):
    def __init__(self, data_path, labels_path, device):
        try:
            if data_path.endswith('.npy'):
                x = np.load(data_path)
            else:
                raise ValueError('data_file should be .npy file')
        except:
            x = torch.load(data_path).float().to(device)
        labels = np.loadtxt(labels_path)
        self.x = torch.Tensor(x).to(device)
        self.labels = torch.FloatTensor(labels).to(device)

    def __getitem__(self, idx):
        return  self.x[idx], self.labels[idx]

    def __len__(self):
        return len(self.x)

    def getinputsize(self):
        return len(self.x[0])

    def getoutputsize(self):
        return  len(self.labels[0])

class RNAbidngDataset(Dataset):
    def __init__(self, data_path, name_list, device=None):
        with open(name_list, 'r') as f:
            prots = f.read().split()
        existed_prots = [file.split('.')[0] for file in os.listdir(data_path)]
        if len(prots) > len(existed_prots):
            prots = existed_prots
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.label_file = "/home2/xeweng/data/RNA_binding/labels.txt"
        label_dict = {}
        with open(self.label_file, 'r') as f:
            for line in f:
                name, label = line.strip().split()
                label_dict[name] = int(label)
        self.label_dict = label_dict

        self.get_inputs(prots, data_path)
        self.get_labels(prots)


    def get_inputs(self, prots, data_path):
        dataset = []
        for prot in prots:
            vector = np.load(os.path.join(data_path, prot+'.npy'))
            dataset.append(vector)
        self.x = torch.tensor(np.vstack(dataset), dtype=torch.float32).to(self.device)
        return self.x

    def get_labels(self, prots):
        labels = []
        for prot in prots:
            labels.append(self.label_dict[prot])
        self.labels = torch.tensor(np.array(labels), dtype=torch.long).to(self.device)
        self.labels = self.labels.unsqueeze(1)
        return self.labels
    
    def __getitem__(self, idx):
        return  self.x[idx], self.labels[idx]

    def __len__(self):
        return len(self.x)

    def getinputsize(self):
        return len(self.x[0])

    def getoutputsize(self):
        return  1

class MLP(nn.Module):
    def __init__(self, feature_num, label_num, hidden_num, num_layers=2, batch_norm=False, dropout=0):
        super().__init__()
        layers = [nn.Linear(feature_num, hidden_num)]
        for i in range(num_layers):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_num))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.Linear(hidden_num, hidden_num))
        layers.append(nn.Linear(hidden_num, label_num))
        self.linear_relu_stack = nn.Sequential(*layers)
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class BinaryClassifier(nn.Module):
    def __init__(self, feature_num, label_num, hidden_num):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_num, hidden_num),
            nn.BatchNorm1d(hidden_num),  # Batch Normalization
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num),
            nn.BatchNorm1d(hidden_num),  # Batch Normalization
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_num, label_num),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)


def train_loop(dataloader, model, loss_fn, optimizer, metric='fmax'):
    num_batchs = len(dataloader)
    size = len(dataloader.dataset)

    model.train()
    loss_value = 0
    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)

        loss = loss_fn(pred, y)
        loss_value += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def valid_loop(dataloader, model, loss_fn, metric='fmax'):
    model.eval()

    num_batches = len(dataloader)
    test_loss = 0

    preds, targets = [], []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()

            preds.append(pred.view(pred.size(0), -1))
            targets.append(y)
            # print('batch loss:', loss.item())
    test_loss /= num_batches
    pred = torch.cat(preds)
    target = torch.cat(targets)
    fmax = f1_max(pred, target)
    print(f"Valid Dataset Result: \n Fmax: {(fmax):>0.7f}, Avg loss: {test_loss:>8f} \n")
    return fmax, loss

def compute_metrics(model, dataloader, device):
    model.eval()
    accuracy = Accuracy(task="binary").to(device)
    precision = Precision(num_classes=1, average='macro', task="binary",multiclass=False).to(device)
    recall = Recall(num_classes=1, average='macro', task="binary", multiclass=False).to(device)

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            preds = (outputs > 0.5).int()
            accuracy.update(preds, y_batch)
            precision.update(preds, y_batch)
            recall.update(preds, y_batch)

    # 计算最终指标
    test_accuracy = accuracy.compute().to(device)
    test_precision = precision.compute().to(device)
    test_recall = recall.compute().to(device)
    return test_accuracy, test_precision, test_recall

def compute_auroc(pred, target):
    roc_auc = AUROC(task="multiclass", num_classes=10)
    return roc_auc(pred, target)

def evaluate(preds, targets):
    mectrics = ['acc', 'roc_auc']
    scores = {}
    for metric in mectrics:
        if metric == 'acc':
            preds = (preds > 0.5).astype(int)
            correct_num = ((preds == targets).sum()).item()
            total_num = len(preds)
            accuracy = correct_num / total_num
            scores[metric] = accuracy
        elif metric == 'roc_auc':
            targets = targets.squeeze()
            preds = preds.squeeze()

            scores[metric] = roc_auc_score(targets, preds)

    return scores


def test_evaluation(dataloader, model, loss_fn, metric='fmax'):
    '''
        args:

    '''
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.
    # fmax = Fmax(dataloader, model)
    preds, targets = [], []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()

            preds.append(pred.view(pred.size(0), -1))
            targets.append(y)

    test_loss /= num_batches
    pred = torch.cat(preds)
    target = torch.cat(targets)

    score = 0
    if metric == 'fmax':
        score = f1_max(pred, target)
    elif metric == 'roc-auc':
        score = compute_auroc(model, dataloader)
    
    return score, loss