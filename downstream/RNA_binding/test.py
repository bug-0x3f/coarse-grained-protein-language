

import torch
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore")
import os
import pickle

from utils import RNAbidngDataset, BinaryClassifier, evaluate


def build_dataset(data_path, len_type, device):
    if len_type == 'all':
        name_list = "/home2/xeweng/data/RNA_binding/test_prot_list.txt"
    else:
        name_list = f'/home2/xeweng/experiment/long_short/data/binary/RNA_binding_test_{len_type}_prots_namelist.txt'
    test_data_path = f"{data_path}/test/sentence_vector/"
    
    return RNAbidngDataset(test_data_path, name_list, device)


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
    task = args.task
    test_data_path = args.data_path
    gpu = args.gpu
    scores_save_dir = args.scores_save_dir
    print(task)
    os.makedirs(scores_save_dir, exist_ok=True)
    
    score_file = f'{scores_save_dir}/{task}.pkl'

    score_dic_list = []
    
    name_list = "./test_prot_list.txt"
    model_path = 'model.pth'
    
    device = torch.device(f'cuda:{gpu}')
    test_dataset = RNAbidngDataset(test_data_path, name_list, device)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    samples_f1 = []
    for i in range(5)[:]:
        input_size = test_dataset.getinputsize() 
        output_size = test_dataset.getoutputsize()  
        hidden_size = 512 
        model = BinaryClassifier(input_size, output_size, hidden_size).to(device)
        
        state_dict = torch.load(f'{model_path}/model_{i}.pth', map_location=device)

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

            scores = evaluate(pred, target)
        

