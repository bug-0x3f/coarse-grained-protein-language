import sys
import os

from utils import RNAbidngDataset, BinaryClassifier

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import wandb
from sklearn.metrics import roc_curve, auc, roc_auc_score
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore")


def test_evaluation(dataloader, model, loss_fn, logits=False):
    '''
        args:
    '''
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.

    preds, targets = [], []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y.float())
            test_loss += loss.item()

            preds.append(pred.view(pred.size(0), -1))
            targets.append(y)

    test_loss /= num_batches
    pred = torch.cat(preds)
    target = torch.cat(targets)

    scores = evaluate(pred, target)
    if logits:
        return scores, loss, pred, target
    else:
        return scores, loss

def main(data_path):
    import random
    gpu = int(sys.argv[1])
    comment = sys.argv[2]
    model_idx =  sys.argv[4]
    model_save_path = sys.argv[5]
    save_dir = f"{model_save_path}/saved_models{model_idx}/{comment}"
    os.makedirs(save_dir, exist_ok=True)

    train_data_path = f'.'
    valid_data_path = f'.'

    train_name_list = "train_prot_list.txt"
    valid_name_list = "valid_prot_list.txt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(gpu)

    batch_size = 32
    train_dataset = RNAbidngDataset(train_data_path, train_name_list, device)

    valid_dataset = RNAbidngDataset(valid_data_path, valid_name_list, device)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    input_size = train_dataset.getinputsize()  # 输入特征的维度
    output_size = train_dataset.getoutputsize()  # 输出的类别数
    hidden_size = 512 # 隐藏层的维度

    print(f'=== comment:{comment} gpus:{device}{gpu} labels: {output_size}', 
          f'inputsize:{input_size} train_data_len:{len(train_dataset)}', 
          f'test_data_len:{len(valid_dataset)}' )

    learning_rate = 5e-4
    model = BinaryClassifier(input_size, output_size, hidden_size).to(device)
    model = torch.nn.DataParallel(model, device_ids=[gpu])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 200
    patience = 20
    counter = 0
    top_k = 5

    loss_fn = nn.BCELoss()


    best_models = []
    best_test_score = -1

    for i in range(num_epochs):
        model.train()
        total_loss = 0

        for X, train_labels in train_dataloader:
            preds = model(X)
            train_loss = loss_fn(preds, train_labels.float())
            total_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


        scores, val_loss, pred, target = test_evaluation(test_dataloader, model, loss_fn, logits=True)
        auroc_score = scores['roc_auc']


        best_test_score = max(auroc_score, best_test_score)                          
        model_info = {
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'val_score': scores,
            'val_loss': val_loss  
        }
        metric = 'acc'
        if len(best_models) < top_k:
            best_models.append(model_info)
        elif scores[metric] > min([m['val_score'][metric] for m in best_models]):  
            worst_idx = min(range(len(best_models)), key=lambda i: best_models[i]['val_score'][metric])
            best_models.pop(worst_idx)
            best_models.append(model_info)
            best_models.sort(key=lambda x: x['val_score'][metric])


        if  scores[metric] < best_test_score:
            counter += 1
            if counter >= patience:
                print(f"early stopping at epoch {i}.")
                break  
        else:
            best_test_score = scores[metric]
            counter = 0  

    print(f"finish task, {comment} gpu:{gpu} best score: {best_test_score}")
   

    model_path = os.path.join(save_dir, f"model.pth")
    torch.save(best_models[-1]['model_state_dict'], model_path)

if __name__ == '__main__':

    main(sys.argv[3])
