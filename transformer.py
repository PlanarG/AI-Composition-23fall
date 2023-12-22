import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sad = np.load('./music-source/sad.npy');
happy = np.load('./music-source/happy.npy');
rnd = np.load('./music-source/rnd.npy');

def convert(comp):
    result = []
    for i in range(comp.shape[0]):
        tmp = []
        for j in range(comp.shape[1] // 2):
            tmp.append(comp[i][j * 2] * 2 + comp[i][j * 2 + 1])
        result.append(tmp)
    return np.array(result)

# MIDI pitch range, used in one-hot encoding
# equals to 2 * pitch + break(0/1) numerically
input_size = 250

# dimension of embedding vector
hidden_size = 64

# number of encoding layers
num_layers = 2

# number of attention heads
num_heads = 2

# number of epochs
num_epochs = 5

###
### Load data
###

# file format: [[2 * pitch + break], [...], ...]
# x_train = np.load('./data/x_train.npy')
x_train = np.concatenate((convert(sad), convert(happy)), axis = 0)

# file format: [sorrowness for song 1, sorrowness for song 2, ...]
# sorrow_train = np.load('./data/sorrow_train.npy')
sorrow_train = np.concatenate((np.array([1.] * sad.shape[0]), 
                               np.array([0.] * happy.shape[0])
                                ), axis=0).astype(np.float32)

print(sad.shape, happy.shape, x_train.shape, sorrow_train.shape)


train_data = TensorDataset(torch.tensor(x_train), torch.tensor(sorrow_train))
train_loader = DataLoader(dataset = train_data, batch_size = 16, shuffle = True)

###
### Define model
###

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        # sorrowness
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer(x, x)
        output = torch.mean(output, dim = 1)
        sorrowness = self.fc(output)
        return sorrowness

model = Transformer(input_size, hidden_size, num_heads, num_layers)

###
### Train model
### 

criterion1 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

losses = []

for epoch in range(num_epochs):
    for x, sorrow_y in tqdm(train_loader):
        optimizer.zero_grad()
        sorrowness = model(x)
        print(sorrowness)
        loss = criterion1(sorrowness, sorrow_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

# torch.save(model, './model/transformer.pt')

###
### Plot loss
###

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over training')
plt.plot(losses)
plt.show()

