import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sad = np.load('./music-source/sad.npy');
happy = np.load('./music-source/happy.npy');
rnd = np.load('./music-source/rnd.npy');

# MIDI pitch range, used in one-hot encoding
input_size = 128

# dimension of embedding vector
hidden_size = 32

# number of encoding layers
num_layers = 1

# number of attention heads
num_heads = 2

# number of epochs
num_epochs = 30

###
### Load data
###

x_data = np.concatenate((sad, happy, rnd), axis = 0)

y_data = np.concatenate((np.array([1.] * sad.shape[0]), 
                        np.array([0.5] * happy.shape[0]),
                        np.array([0.] * rnd.shape[0])), axis=0).astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

train_data = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
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
        self.fc = nn.Linear(hidden_size, 1)
    
    
    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer(x, x)
        output = torch.mean(output, dim = 1)
        sorrowness = self.fc(output)
        return torch.sigmoid(sorrowness)

model = Transformer(input_size, hidden_size, num_heads, num_layers)

model = torch.load('./model/transformer.pt')

###
### Train model
### 

criterion1 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

losses = []

for epoch in range(num_epochs):
    model.train()
    for x, sorrow_y in tqdm(train_loader):
        optimizer.zero_grad()
        sorrowness = model(x)
        sorrow_y = torch.Tensor(np.expand_dims(sorrow_y, -1))
        loss = criterion1(sorrowness, sorrow_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        y = model(torch.tensor(x_test))
        loss = criterion1(y, torch.Tensor(np.expand_dims(y_test, -1)))
        print(loss.item())
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

