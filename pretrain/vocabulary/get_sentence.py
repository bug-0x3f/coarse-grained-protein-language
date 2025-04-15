
from model import VQVAEWithAttention
import torch
import numpy as np
import os
import argparse, yaml
from easydict import EasyDict
from tqdm import tqdm


cnt = [0] * 2000


def get_and_save_indexs(filepath, save_path):
    sentence_file = os.path.join(save_path, filepath.split('/')[-1])
    
    if filepath.endswith('.npy'):
        x = np.load(filepath)
    else:
        raise ValueError("File format not supported. Please provide a .npy file.")

    x = torch.Tensor(x).to(device)

    if x.dim() == 1:
        x = x.unsqueeze(0)

    res = model.get_index(x)
    res = res.tolist()
    for item in res:
        cnt[item] += 1
    
    with open(sentence_file, 'w') as f:
        for item in res:
            f.write(str(item) + ' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--featurepath', '-f', type=str, default="", required=True, help="a path of input feature files")
    parser.add_argument('--sentence_path', '-s', type=str, default="", required=True,
                        help="")
    parser.add_argument('--name_list', '-l', type=str, default="", required=False)
    parser.add_argument('--model_config', '-c', required=True)
    args = parser.parse_args()
    feature_path = args.featurepath
    

    with open(args.model_config, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))
    # load data
    batch_size = config.trainer.batch_size
    num_epochs = config.trainer.num_epochs
    input_dim = config.model.input_dim
    hidden_dim = config.model.hidden_dim 
    codebook_size = config.model.codebook_size
    embedding_dim = config.model.embedding_dim
    dict_path = config.model.save_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.cuda.set_device(2)

    model = VQVAEWithAttention(input_dim, hidden_dim, embedding_dim, codebook_size, 10, 1024).to(device)
    model.eval()
    model.load_state_dict(torch.load(dict_path))
    sentences_path = args.sentence_path
    
    print(sentences_path)
    files = os.listdir(feature_path)

    if args.name_list != '':
        with open(args.name_list, 'r') as f:
            prot_list = f.read().strip().split()
        files = [prot +'.npy' for prot in prot_list]

    if not os.path.exists(sentences_path):
        os.makedirs(sentences_path)
    for file in tqdm(files):
        get_and_save_indexs(os.path.join(feature_path, file), sentences_path)
    
    sum = 0
    for item in cnt:
        if item > 0:
            sum += 1
    print('codebook usage', sum, codebook_size)
    
