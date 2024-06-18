import torch
from torch import nn
import random 
from torch.utils.data import Dataset


class RecDataset(Dataset):
    def __init__(self, users, items, item_per_users):
        self.users = users
        self.items = items
        self.item_per_users=item_per_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        user = self.users[i]
        return torch.tensor(user), torch.tensor(self.items[i]), self.item_per_users[user]


class PointWiseFFN(nn.Module):
    def __init__(
        self,
        args
    ):
        super(PointWiseFFN, self).__init__()
        self.w_1 = torch.nn.Linear(args.hidden_dim, args.hidden_dim)
        self.relu = torch.nn.ReLU()
        self.w_2 = torch.nn.Linear(args.hidden_dim, args.hidden_dim)
        self.dropout = torch.nn.Dropout(p=args.dropout)
    
    def forward(self, inputs):
        outputs = self.w_1(inputs)
        outputs = self.relu(outputs)
        outputs = self.w_2(outputs)
        outputs = self.dropout(outputs)
        
        return outputs

class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        args
    ):
        super(AttentionBlock, self).__init__()
        
        self.attn_layernorm = torch.nn.LayerNorm(args.hidden_dim,
                                                 eps=1e-8)
        self.attn_layer = torch.nn.MultiheadAttention(args.hidden_dim, args.n_heads,
                                                      dropout=args.attention_dropout,
                                                      batch_first=True)
        self.forward_layernorm = torch.nn.LayerNorm(args.hidden_dim,
                                                    eps=1e-8)
        self.forward_layer= PointWiseFFN(args)
    
    def forward(self, inputs, attn_mask):
        outputs = inputs
        Q = self.attn_layernorm(outputs)
        outputs += self.attn_layer(Q, outputs, outputs,
                                   attn_mask=attn_mask)[0] # Residual Connection
        outputs += self.forward_layer(self.forward_layernorm(outputs)) # Residual Connection
        
        return outputs

class LatentFactorModel(nn.Module):
    def __init__(self, edim, user_indexes, node_indexes):
        super(LatentFactorModel, self).__init__()
        self.edim = edim
        self.users = nn.Embedding(max(user_indexes) + 1, edim)
        self.items = nn.Embedding(max(node_indexes) + 1, edim)

    def forward(self, users, items):
        user_embedings = self.users(users).reshape(-1, self.edim )
        item_embedings = self.items(items)
        res = torch.einsum('be,bne->bn', user_embedings, item_embedings)
        return res 

    def pred_top_k(self, users, K=10):
        user_embedings = self.users(users).reshape(-1, self.edim )
        item_embedings = self.items.weight
        res = torch.einsum('ue,ie->ui', user_embedings, item_embedings)
        return torch.topk(res, K, dim=1)

class SASRec(nn.Module):
    def __init__(self, edim, user_indexes, node_indexes, max_seq_len, num_heads, num_layers, dropout_rate):
        super(SASRec, self).__init__()
        self.edim = edim
        self.max_seq_len = max_seq_len
        
        self.users = nn.Embedding(max(user_indexes) + 1, edim)
        self.items = nn.Embedding(max(node_indexes) + 1, edim)
        
        self.position_embedding = nn.Embedding(max_seq_len, edim)
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=edim, 
                nhead=num_heads, 
                dim_feedforward=edim * 4, 
                dropout=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(edim)

    def forward(self, users, item_sequences):
        user_embeddings = self.users(users).reshape(-1, 1, self.edim)
        seq_embeddings = self.items(item_sequences)
        
        # Ensure the positions tensor is the same length as the item sequences
        seq_len = item_sequences.size(1)
        positions = torch.arange(seq_len, device=item_sequences.device).unsqueeze(0).expand(item_sequences.size(0), -1)
        position_embeddings = self.position_embedding(positions)
        
        seq_embeddings += position_embeddings
        seq_embeddings = self.dropout(seq_embeddings)
        
        for transformer in self.transformer_blocks:
            seq_embeddings = transformer(seq_embeddings)
        
        seq_embeddings = self.layer_norm(seq_embeddings)
        
        # Using the last item in the sequence for prediction
        final_seq_embedding = seq_embeddings[:, -1, :]
        
        return final_seq_embedding
        
    def pred_top_k(self, users, item_sequences, K=30):
        final_seq_embedding = self.forward(users, item_sequences)
        item_embeddings = self.items.weight
        res = torch.einsum('ue,ie->ui', final_seq_embedding, item_embeddings)
        return torch.topk(res, K, dim=1)

def collate_fn(batch, num_negatives, num_items):
    users, target_items, users_negatives = [],[], []
    for triplets in batch:
        user, target_item, seen_item = triplets
        
        users.append(user)
        target_items.append(target_item)
        user_negatives = []
        
        while len(user_negatives)< num_negatives:
            candidate = random.randint(0, num_items)
            if candidate not in seen_item:
                user_negatives.append(candidate)
                
        users_negatives.append(user_negatives)
                
    
    positive = torch.ones(len(batch), 1)       
    negatives = torch.zeros(len(batch), num_negatives)
    labels = torch.hstack([positive, negatives])
    # print(torch.tensor(target_items))
    # print(users_negatives)
    items = torch.hstack([torch.tensor(target_items).reshape(-1, 1), torch.tensor(users_negatives)])
    return torch.hstack(users), items, labels