

from operator import sub
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import copy
import sys
sys.path.append('..')
from utils import Fmax, f1_max, GoAnnots


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


def test_evaluation(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.
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
    fmax = f1_max(pred, target)
    
    return fmax, loss

import sys
def main():

    s, d = sys.argv[1:3]
    subtask = s
    gpus = [int(d)]
    comment = sys.argv[3]
    data_path = sys.argv[4]    
 
    train_data_path = f'{data_path}/train.npy' # set train data
    valid_data_path = f'{data_path}/test.npy'
    train_labels = f"{data_path}/{subtask}_train_target.npy"
    valid_labels = f"{data_path}/{subtask}_valid_target.npy"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(gpus[0])

    bacth_size = 2
    train_dataset = GoAnnots(train_data_path, train_labels, device)
    valid_dataset = GoAnnots(valid_data_path, valid_labels, device)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bacth_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=bacth_size, shuffle=False)

    input_size = train_dataset.getinputsize()  
    output_size = train_dataset.getoutputsize() 
    hidden_size = output_size 
    print(f'=== comment:{comment} subtask:{subtask} gpus:{device}{gpus}', 
          f'inputsize:{input_size} output:{output_size} train_data_len:{len(train_dataset)}', 
          f'test_data_len:{len(valid_dataset)}' )
  

    learning_rate = 0.0001
    num_epochs = 200
    model = MLP(input_size, output_size, hidden_size).to(device)
    model = torch.nn.DataParallel(model, device_ids=gpus)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    save_dir = f"{sys.argv[6]}/saved_models{sys.argv[5]}/{comment}"

    loss_fn = nn.BCEWithLogitsLoss()

    best_test_score = -1
    best_models = []
    best_val_loss = 10
    patience = 10
    # 训练
    counter = 0
    top_k = 5
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()

        total_loss = 0
        for X, train_labels in train_dataloader:
            logits = model(X)
            loss = loss_fn(logits, train_labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        fmax, val_loss = test_evaluation(valid_dataloader, model, loss_fn)
        best_test_score = max(fmax, best_test_score)                          
        model_info = {
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'val_score': fmax,
            'val_loss': val_loss,  
            'epoch':epoch
        }

        if len(best_models) < top_k:
            best_models.append(model_info)
        elif  fmax > min([m['val_score'] for m in best_models]): 
            worst_idx = min(range(len(best_models)), key=lambda i: best_models[i]['val_score'])
            best_models.pop(worst_idx)
            best_models.append(model_info)
            best_models.sort(key=lambda x: x['val_score'])
 
        if  fmax < best_test_score:
            counter += 1
            if counter >= patience:
                print(f"{subtask} early stopping at epoch {epoch}.")
                break  
        else:
            best_test_score = fmax
            counter = 0  #
   

    print(f"finish {subtask} task, comment:{comment} gpu:{gpus} best score: {best_test_score:>0.4f}")

    os.makedirs(save_dir, exist_ok=True)
    print('model_idx', sys.argv[5])

    model_path = os.path.join(save_dir, f"{sub}_model.pth")
    torch.save(best_models[-1]['model_state_dict'], model_path)

if __name__ == '__main__':
    main()
