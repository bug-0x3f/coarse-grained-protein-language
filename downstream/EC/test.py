
import time
from requests import get
import torch
from torch import dtype, nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import sys
sys.path.append('..')
from utils import f1_max


class EcDataset(Dataset):
    def __init__(self, data_path, labels_path, name_list, device):
        self.device = device

        with open(name_list, 'r') as f:
            prots = f.read().split()

        self.get_inputs(prots, data_path)
        self.get_labels(prots, labels_path)

    def __getitem__(self, idx):
        return  self.x[idx], self.labels[idx]

    def __len__(self):
        return len(self.x)

    def getinputsize(self):
        return len(self.x[0])

    def getoutputsize(self):
        return  len(self.labels[0])
    
    def get_inputs(self, prots, data_path):
        dataset = []
        for prot in prots:
            try:
                vector = np.load(os.path.join(data_path, prot+'.npy'))
            except:
                vector = np.loadtxt(os.path.join(data_path, prot+'.txt'))
            dataset.append(vector)
        self.x = torch.tensor(np.vstack(dataset), dtype=torch.float32).to(self.device)
        return self.x

    def get_labels(self, prots, labels_path):
        labels = []
        for prot_name in prots:
            label = np.load(os.path.join(labels_path, prot_name + '.npy'))
            labels.append(label)
        self.labels = torch.tensor(np.vstack(labels), dtype=torch.float32).to(self.device)
        return self.labels

class MLP(nn.Module):
    def __init__(self, feature_num, label_num, hidden_num):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_num, label_num),
            nn.ReLU(),
            nn.Linear(label_num, label_num),
            nn.ReLU(),
            nn.Linear(label_num, label_num),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default='ours')
    parser.add_argument('--data_path', '-v', type=str, default='')
    parser.add_argument('--scores_save_dir', '-s', type=str, default='')
    parser.add_argument('--model_path', '-m', type=str, default='')
    parser.add_argument('--gpu', '-g', type=int, default=4)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    data_path = args.data_path
    gpu = args.gpu
    scores_save_dir = args.scores_save_dir
    os.makedirs(scores_save_dir, exist_ok=True)


    score_dic_list = []
    labels_path = ''
    name_list = ''
    model_path = f"{args.model_path}/saved_models/"
    score_dic = {}

            
    device = torch.device(f'cuda:{gpu}')
    test_dataset = EcDataset(data_path, labels_path, name_list, device)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    scores = []

    for i in range(5)[:]:
        input_size = test_dataset.getinputsize()  
        output_size = test_dataset.getoutputsize()  
        hidden_size = output_size 
        model = MLP(input_size, output_size, hidden_size).to(device)
        
        state_dict = torch.load(f'{model_path}/model.pth', map_location=device)
        model = torch.nn.DataParallel(model, device_ids=[gpu])
        model.load_state_dict(state_dict)

        model.eval()
        with torch.no_grad():
            all_f1 = []
            preds, targets = [], []
            for x, y in test_loader:
                pred = model(x)
                preds.append(pred)
                targets.append(y)
            pred = torch.cat(preds)
            target = torch.cat(targets)

            f1 = f1_max(pred, target)
            scores.append(f1)



