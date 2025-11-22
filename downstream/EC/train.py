import sys
from utils import *
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
import os

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
    score = f1_max(pred, target)
    
    return score, loss

def main(data_path):

    gpu = int(sys.argv[1])
    comment = sys.argv[2]

    model_idx = sys.argv[4]

    save_dir = f"{sys.argv[5]}/saved_models{model_idx}/{comment}"
    os.makedirs(save_dir, exist_ok=True)

    train_data_path = f'{data_path}/train.npy'
    valid_data_path = f'{data_path}/valid.npy'
    train_labels = f"{data_path}/train_labels.npy"
    valid_labels = f"{data_path}/test_labels.npy"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(gpu)

    bacth_size = 2
    train_dataset = GoAnnots(train_data_path, train_labels, device)
    valid_dataset = GoAnnots(valid_data_path, valid_labels, device)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bacth_size, shuffle=True)
    test_dataloader = DataLoader(dataset=valid_dataset, batch_size=bacth_size, shuffle=False)

    input_size = train_dataset.getinputsize()
    output_size = train_dataset.getoutputsize()
    hidden_size = output_size


    print(f'=== comment:{comment} gpus:{device}{gpu} labels: {output_size}', 
          f'inputsize:{input_size} train_data_len:{len(train_dataset)}', 
          f'test_data_len:{len(valid_dataset)}' )


    learning_rate = 0.0001
    model = MLP(input_size, output_size, hidden_size).to(device)
    model = torch.nn.DataParallel(model, device_ids=[gpu])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    num_epochs = 200
    loss_fn = nn.BCEWithLogitsLoss()
    best_test_score = -1
    best_models = []
    patience = 10
    top_k = 5
    counter = 0
    # 训练
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


        fmax, val_loss = test_evaluation(test_dataloader, model, loss_fn)
        best_test_score = max(fmax, best_test_score)                          
        model_info = {
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'val_score': fmax,
            'val_loss': val_loss  
        }
        if len(best_models) < top_k:
            best_models.append(model_info)

        elif fmax > min([m['val_score'] for m in best_models]): 
            worst_idx = min(range(len(best_models)), key=lambda i: best_models[i]['val_score'])
            best_models.pop(worst_idx)
            best_models.append(model_info)
            best_models.sort(key=lambda x: x['val_score'])

        if  fmax < best_test_score:
            counter += 1
            if counter >= patience:
                print(f" early stopping at epoch {epoch}.")
                break  
        else:
            best_test_score = fmax
            counter = 0  

    print(f"finish task, comment:{comment} best score: {best_test_score}")

    
    model_path = os.path.join(save_dir, f"model.pth")
    torch.save(best_models[-1]['model_state_dict'], model_path)

if __name__ == '__main__':

    main(sys.argv[3])
