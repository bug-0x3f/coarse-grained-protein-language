import torch
import time
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = self.load_data(dataset)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        return sample
    
    def load_data(self, dataset):
        # Load and preprocess the dataset
        import numpy as np
        start_time = time.time()
        if dataset.endswith('.npy'):
            data = np.load(dataset)
        else:
            raise ValueError("File format not supported. Please provide a .npy file.")
        data = torch.FloatTensor(data)
        print('数据加载完成', data.shape, 'need time', time.time()-start_time)
        
        return data
    