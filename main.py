import torch
from torch.nn import functional as F
from torch import nn
from model import Model
from tqdm import tqdm

context_size = 10
n_embed = 32
n_hidden = 128
xlen = 100
rands = torch.randn(xlen)

def break_contexts(rands):
    ix = torch.arange(len(rands) - context_size)
    x = torch.stack([rands[i:i+context_size] for i in ix])
    y = torch.stack([torch.tensor([rands[i+context_size]]) for i in ix])
    
    return list(zip(x, y))

dataloader = break_contexts(rands)

model = Model(context_size, n_embed, n_hidden)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

print("before training position encodings:")
print(model.pos_encoder.positions)
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        # print(outputs, y)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(x):.4f}")
print("After training position encoding:")
print(model.pos_encoder.positions)
