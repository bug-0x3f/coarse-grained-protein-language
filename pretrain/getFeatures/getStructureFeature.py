import imp
import random
from networkx import to_edgelist
from sympy import Q
from structure.model import LSTMModel
import torch

gpus = [ 0, 3, 4, 7]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(random.choice(gpus)) # 分散到各个gpu
# print('device:', device)

input_dim = 4 
hidden_dim = 32
layer_dim = 2
output_dim = 4 
learning_rate = 0.0001

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
model_path = "/home2/xeweng/data/nlp_model/trained_model_lr0.001.pt"
model_path = f"/home2/xeweng/data/nlp_model/trained_model_bs(128)_lr(0.001)_hd({hidden_dim})_ep200.pt"

# print(model_path)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()
from structure.getFragInform import get_frags_inform

def get_structure_feature(dsspfile, filter_flag=True):
    import numpy as np
    # 提取坐标
    
    try:
        frags_infom, labels = get_frags_inform(dsspfile, filter_flag=filter_flag)
         
    except:
        print(os.path.join(dssp_path, dssp))
        return
    # print(len(h), len(frags_inform))
    max_len = 60
    for index, frag in enumerate(frags_inform):
        # print(type(frag))
        while len(frag) < max_len:                                
            frag.append([0]*4)

        if len(frag) > 60:
            # 随机截取60
            cut_index = random.randint(0, len(frag) - max_len)
            frags_inform[index] = frag[cut_index: cut_index + max_len]
    
    s = torch.tensor(frags_inform)
    ret = []
    # print('---------')
    try:
        ret = model.get_sentences_feature(s.to(torch.float32))
    except Exception as e:
        print('get lstm structure feature error', e)
        print(s.shape)
    # for sentence in frags_info:
    #     # 检查sentence的类型
    #     if sentence == []:
    #         ret.append(np.array([0]*hidden_dim))
    #     else:
    #         # print(sentence, len(sentence))
    #         sentence = torch.Tensor(np.array(sentence)).to(device)
    #         # print('sentence shape', sentence.shape)
    #         try:
    #             ret.append(model.get_sentence_feature(sentence).detach().cpu().numpy())
    #             # print('0kk')
    #         except Exception as e:
    #             print('get lstm structure feature error', e)
    #             print(sentence.shape)
    #             break
    
    return np.array(ret)

def main(dssp_path):
    pool = Pool(1)
    # 使用列表推导式获取文件夹下所有文件的绝对路径列表
    file_paths = [os.path.join(root, file_name) for  file_name in os.listdir(dssp_path)]
    pool.map(get_structure_feature, file_paths)
    pool.close()
    pool.join()

def script():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dssp', help='dssp文件夹路径')
    args = parser.parse_args()
    main(args.input, args.dssp)

if __name__ == "__main__":
    pass
    # 构建只有结构特征的数据集