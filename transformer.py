import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

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
num_epochs = 100

###
### Load data
###

# file format: [[2 * pitch + break], [...], ...]
x_train = np.load('./data/x_train.npy')

# file format: [sorrowness for song 1, sorrowness for song 2, ...]
sorrow_train = np.load('./data/sorrow_train.npy')

# file format: [rythmness for song 1, rythmness for song 2, ...]
rythmn_train = np.load('./data/rythmn_train.npy')

train_data = TensorDataset(x_train, sorrow_train, rythmn_train)
train_loader = DataLoader(dataset = train_data, batch_size = 32, shuffle = True)

###
### Define model
###

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = Transformer(
            d_model=hidden_size,
            num_heads=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        # sorrowness
        self.fc1 = nn.Linear(hidden_size, 1)
        # rythmness
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer(x, x)
        output = torch.mean(output, dim = 1)
        sorrowness = self.fc1(output)
        rythmness = self.fc2(output)
        return sorrowness, rythmness

model = Transformer(input_size, hidden_size, num_heads, num_layers)

###
### Train model
### 

criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

losses = []

for epoch in range(num_epochs):
    for x, sorrow_y, rythmn_y in train_loader:
        optimizer.zero_grad()
        sorrowness, rythmness = model(x)
        loss = criterion1(sorrowness, sorrow_y) + criterion2(rythmness, rythmn_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

torch.save(model, './model/transformer.pt')

###
### Plot loss
###

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over training')
plt.plot(losses)
plt.show()

