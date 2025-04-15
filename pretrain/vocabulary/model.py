import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_heads, intermediate_size):
        super(AttentionEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)  # Project input to embedding dim
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)  # Normalization after attention
        self.mlp = nn.Sequential(            
            nn.Linear(hidden_dim, intermediate_size),
            # nn.BatchNorm1d(hidden_dim),  # Batch Normalization
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.codebook_projection = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x):
        x = self.input_projection(x)  
        

        x = x.unsqueeze(1)  # Add a sequence dimension (seq_len=1)

        # Self-attention
        attn_output, _ = self.attention(x, x, x)  # Query, Key, Value are all x
        x = self.norm(attn_output + x)  # Residual connection and normalization

        # Feedforward network (MLP)
        x = self.mlp(x) + x  # Residual connection
        x = self.codebook_projection(x)

        return x.squeeze(1)

    
class VQVAEWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, codebook_size, num_heads, intermediate_size, init_embeddings=None):
        super(VQVAEWithAttention, self).__init__()

        # encoder
        self.encoder = AttentionEncoder(input_dim, hidden_dim, embedding_dim, num_heads, hidden_dim)
        # self.encoder = AttentionEncoder2(input_dim, hidden_dim, embedding_dim, num_heads, intermediate_size) # new
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)  # Initialization
        self.beta = 0.25

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        encoded_output = self.encoder(x)

        dist = torch.cdist(encoded_output, self.embedding.weight)

        min_encoding_indices = torch.argmin(dist, dim=-1)
        quant_out = self.embedding(min_encoding_indices)

        commitment_loss = F.mse_loss(quant_out.detach(), encoded_output)
        codebook_loss = F.mse_loss(quant_out, encoded_output.detach())
        quantize_losses = self.beta * commitment_loss + codebook_loss

        quant_out = encoded_output + (quant_out - encoded_output).detach()   

        output = self.decoder(quant_out)

        return output, quantize_losses

    def get_index(self, x):
        encoded_output = self.encoder(x)
        dist = torch.cdist(encoded_output, self.embedding.weight)
        min_encoding_indices = torch.argmin(dist, dim=-1)
        return min_encoding_indices
    