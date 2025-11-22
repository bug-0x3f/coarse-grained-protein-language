import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os, argparse
from utils import MyDataset
from model import *
from easydict import EasyDict
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
    parser.add_argument('-m', '--comment', help="running configurations", type=str, required=False)
    return parser.parse_args()

def load_config():
    pass

if __name__ == "__main__":
    args = get_args()
    comment = args.comment
    with open(args.config, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))

    print()
    os.makedirs(os.path.dirname(config.model.save_path), exist_ok=True)

    for k, v in config.setting.os_environ.items():
        if v is not None and k not in os.environ:
            os.environ[k] = str(v)

        elif k in os.environ:
            # override the os environment variables
            config.setting.os_environ[k] = os.environ[k]

    dataset = config.dataset.filepath
    intermediate_size = config.model.intermediate_size if 'intermediate_size' in config.model else config.model.hidden_dim
    if config.setting.logger:
        wandb.init(project=config.wandb_config.project, name=config.wandb_config.name,
                    config={
                        "learning_rate": config.trainer.learning_rate,
                        "batch_size": config.trainer.batch_size,
                        "epochs": config.trainer.num_epochs,
                        'hidden_size': config.model.hidden_dim,
                        'codebook_size': config.model.codebook_size,
                        'embedding_dim': config.model.embedding_dim,
                        'intermediate_size': intermediate_size,
                        'seed_state': torch.get_rng_state()
                    })
    

    dataset = MyDataset(dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = config.trainer.batch_size
    num_epochs = config.trainer.num_epochs
    input_dim = config.model.input_dim
    hidden_dim = config.model.hidden_dim # encoder hidden size
    codebook_size = config.model.codebook_size
    embedding_dim = config.model.embedding_dim

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = VQVAEWithAttention(input_dim, hidden_dim, embedding_dim, codebook_size, num_heads=10, intermediate_size=intermediate_size)
    model = model.to(device)

    criterion = nn.KLDivLoss(reduction='batchmean')

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.trainer.learning_rate)
    
    from datetime import datetime
    

    print('start training')
    
    # Training loop
    num_batches = len(train_dataloader)
    for epoch in range(num_epochs + 1):
        start_time = datetime.now()  
        total_loss = 0
        total_loss1 = 0
        total_qloss = 0
        
        if epoch == 0:
            model.eval()
        else:
            model.train()

        for batch in train_dataloader:

            batch = batch.to(device)
            optimizer.zero_grad()
            output, quantize_losses = model(batch)

            kl_loss = criterion(F.log_softmax(output, dim=-1), F.softmax(batch, dim=-1))

            loss = kl_loss + quantize_losses 
            total_loss += loss.item()
            total_loss1 += kl_loss.item()
            total_qloss += quantize_losses.item()

            if epoch > 0:          
                loss.backward()
                optimizer.step()
        if config.setting.logger:
            wandb.log({"total_loss": total_loss / num_batches, 
                    "kl_loss": total_loss1 / num_batches, 
                    "quantize_losses": total_qloss / num_batches})
        
        end_time = datetime.now()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss / num_batches}, Duration: {epoch_duration}")

    
    torch.save(model.state_dict(), config.model.save_path)
    print('model saved, ', config.model.save_path)



