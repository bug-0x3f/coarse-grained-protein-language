
from calendar import c
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import sys
sys.path.append('..')
from utils import f1_max

class TestDataset(Dataset):
    def __init__(self, data_path, labels_path, device):
        self.device = device
       
        namelist = f'../../dataset/GO/test_prots.txt'
        with open(namelist, 'r') as f:
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
            vector = np.load(os.path.join(data_path, prot+'.npy'))
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
    def __init__(self, feature_num, label_num):
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default='ours')
    parser.add_argument('--vecter_path', '-v', type=str, default='')
    parser.add_argument('--scores_save_dir', '-s', type=str, default='')
    parser.add_argument('--model_path', '-m', type=str, default='')
    parser.add_argument('--gpu', '-g', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == "__main__":


    args = get_args()
    task = args.task
    data_path = args.vecter_path
    gpu = args.gpu
    scores_save_dir = args.scores_save_dir
    print(task)


    os.makedirs(scores_save_dir, exist_ok=True)

    score_dic_list = []
    
    model_path = f"{args.model_path}/saved_models"
    score_dic = {}
    test_labels_path = f"" # label_path
    test_data_path = data_path # sentence vector

    for subtask in ['BP','MF', 'CC' ][:]:
        device = torch.device(f'cuda:{gpu}')
        test_dataset = TestDataset(test_data_path, test_labels_path, device)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        scores = []
        samples_f1 = []
        for i in range(5)[:]:
            input_size = test_dataset.getinputsize()  
            output_size = test_dataset.getoutputsize()  
            hidden_size = output_size 
            model = MLP(input_size, output_size, hidden_size).to(device)
            
            state_dict = torch.load(f'{model_path}/{subtask}_model.pth', map_location=device)

            model = torch.nn.DataParallel(model, device_ids=[gpu])
            model.load_state_dict(state_dict)

            model.eval()
            with torch.no_grad():
                preds, targets = [], []
                for x, y in test_loader:
                    pred = model(x)
                    preds.append(pred)
                    targets.append(y)
                pred = torch.cat(preds)
                target = torch.cat(targets)
                fmax = f1_max(pred, target)


