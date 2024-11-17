import torch 
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

head_size = 16
n_embed = 32
block_size = 8
batch_size = 64
learning_rate = 0.00015
n_iters = 10000
n_heads = 8

# Reading and splittin the data into training and validation

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda string: [stoi[ch] for ch in string]
decode = lambda string: ''.join([itos[ch] for ch in string])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9* len(data))
train = data[:n]
validation = data[n:]

# Defining the model

class Head(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.register_buffer('head_size', torch.tensor(head_size))
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x)  # (B,T,C)
        wei = q @ k.transpose(-2,-1) # transpose the last two dimensions (B, T, 16) @ (B, 16, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) * (head_size ** -0.5)
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads*head_size, n_embed)
    
    def forward(self,x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        
        super().__init__()
        # each token reads off the logits for the next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.proj = nn.Linear(head_size, n_embed)
        self.sa_head = Head(block_size=block_size // n_heads)
        self.ma_head = MultiHeadAttention(block_size=block_size)
        
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(idx.shape[1]))
        x = token_embeddings + position_embeddings
        x = self.ma_head(x) # apply self attention head here
        # x = self.proj(x)
        logits = self.lm_head(x)
        
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape # Reshaping

            logits = logits.view(B*T,C)

            targets = targets.view(B*T)

            loss = F.cross_entropy(logits,targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -block_size:]
            
            # predictions
            logits, loss = self(idx_cond)
            # focus only on the last step -> bigram
            logits = logits[:,-1,:] # Makes it (B,C)
            # apply softmax
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx

idx = torch.zeros((1,1), dtype=torch.long)
m = BigramLanguageModel(vocab_size)

def get_batch(split):
    data = train if split == 'train' else validation
    ix = torch.randint(len(data) - block_size, (batch_size,)) # randomly generated 4 numbers b/w 0 and block size
    x = torch.stack([data[i:i+block_size] for i in ix]) # get chunks for all of the ix
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # get the target for the tensors
    return x,y

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for steps in range(n_iters):
    xb, yb = get_batch('train')
    
    logits, loss = m(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    optimizer.step()
    
    if steps % 1000 == 0:
        print(f"Step {steps}, Loss: {loss.item()}")

print(loss.item())
print(decode(m.generate(idx, max_new_tokens = 1000)[0].tolist()))