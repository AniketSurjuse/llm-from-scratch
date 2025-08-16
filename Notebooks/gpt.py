import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class GPTDatasetV1(Dataset):

    def __init__(self, text, tokenizer,max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)
    

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(text, batch_size, max_length, stride, shuffle = True, drop_last =True, num_workers = 0):

    tokenizer = tiktoken.get_encoding('gpt2')

    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader



class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias= False):
        super().__init__()

        assert d_out% num_heads == 0, 'd_out must be divisible by num_heads'

        #this d_out is larger than the above implementation, it's actually d_out*num_heads of old implementation
        self.d_out = d_out
        self.num_heads = num_heads

        self.head_dim = d_out//  num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, n_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #view them to add extra dimension
        keys = keys.view(b,n_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b,n_tokens, self.num_heads, self.head_dim)
        values = values.view(b,n_tokens, self.num_heads, self.head_dim)

        #transpose to get (b, num_heads, n_tokens, head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #attn_score
        attn_scores = queries @ keys.transpose(2,3)

        #masked attn_score
        attn_scores.masked_fill_(
            self.mask.bool()[:n_tokens, :n_tokens],
            -torch.inf
        )

        #attn_weight
        attn_weights = torch.softmax(
            attn_scores/ keys.shape[-1]**0.5 , dim = -1
        )

        attn_weights = self.dropout(attn_weights)

        context_vecs = attn_weights @ values

        context_vecs = context_vecs.transpose(1,2)

        context_vecs = context_vecs.contiguous().view(b, n_tokens, self.d_out)

        context_vecs = self.out_proj(context_vecs)

        return context_vecs
    


class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim =-1, keepdim = True, unbiased =False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return norm_x
    

class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5* x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0/ torch.pi)) *
            (x + 0.044715 * torch.pow(x,3))
        ))
    

class FeedForward(nn.Module):

    def __init__(self,cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'],4*cfg['emb_dim'] ),
            GELU(),
            nn.Linear(4*cfg['emb_dim'], cfg['emb_dim'])
        )


    def forward(self,x):
        return self.layers(x)
    

class TransfomerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout = cfg['drop_rate'],
            qkv_bias = cfg['qkv_bias']

        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    

class GPT(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransfomerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias = False)

    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device = in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

        

