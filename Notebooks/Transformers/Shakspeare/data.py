
import torch
import torch.nn as nn
from torch.nn import functional as F


#hyper-params

batch_size = 32  #how many of the blocks are you looking at simultaneously
block_size = 8 # To follow the video, we use block sizes/contexts of 8. Each block actually contains 8 different context examples
max_iterations = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)


with open('input.txt', 'r') as f:
    text = f.read()

#read data
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {char:i for i, char in enumerate(chars)}
int_to_char = {i:char for i, char in enumerate(chars)}
encode = lambda string : [char_to_int[char] for char in string]
decode = lambda integers : ''.join([int_to_char[i] for i in integers]) #''.join() joins together characters with nothing between them
data = torch.tensor(encode(text), dtype=torch.long)

#split data
train_val_split = .9 #take 90% val
train_data = data[:int(len(data) * train_val_split)]
val_data = data[int(len(data) * train_val_split):]

#estimate loss func
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#batch load func
def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_start_index = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in batch_start_index])
    y = torch.stack([data[i+1: i + block_size+1] for i in batch_start_index])
    x,y = x.to(device), y.to(device)
    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) #todo understand

    def forward(self, index, targets = None):
        logits = self.token_embedding_table(index) #todo understand

        if targets is not None:
            B,T,C = logits.shape #todo understand
            logits = logits.view(B*T, C) #todo understand
            targets= targets.view(B*T) #todo understand
            loss= F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index) #todo understand
            logits = logits[:, -1, :] #todo understand
            probs = F.softmax(logits, dim=-1) #todo understand
            index_next = torch.multinomial(probs, num_samples=1) #todo understand
            index = torch.cat([index, index_next], dim=1) #todo understand
        return index


model = BigramLanguageModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

for steps in range(max_iterations):
    #eval loss
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {steps}, train loss: {losses['train']}, val loss: {losses['val']}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context= torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
