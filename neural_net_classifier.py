import time
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Fix Random seed
torch.manual_seed(32)

# Process Data 
train = pd.read_csv('data/genre_train.csv')
test = pd.read_csv('data/genre_test.csv')
genres = np.unique(train['label'])
map_genres = {genre: i for i, genre in enumerate(genres)}
train.replace({"label":map_genres},inplace=True)
test.replace({"label":map_genres},inplace=True)

class Data(Dataset):  
    def __init__(self, data):
        self.X = torch.from_numpy(data.drop('label',axis=1).to_numpy()).type(torch.float)
        self.y = torch.from_numpy(data['label'].to_numpy()).type(torch.long)
        self.len = self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.y[index]  
    def __len__(self):
        return self.len
    
traindata = Data(train)
testdata = Data(test)
trainloader = DataLoader(traindata, batch_size=10,shuffle=True, num_workers=0)

# Model Architecture
input_dim = traindata.X.shape[1]
hidden_layers = 8
output_dim = len(genres)
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, hidden_layers)
        self.linear3 = nn.Linear(hidden_layers, hidden_layers)
        self.linear4 = nn.Linear(hidden_layers, output_dim)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    
# Initialize Model
genre_clsfr = Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(genre_clsfr.parameters(), lr=0.01)

# Learning Rate
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Train Model 
start = time.time()
epochs = 100
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # zero parameter gradients 
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = genre_clsfr(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if i % 500 == 499:    # print every 500 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
        #     running_loss = 0.0
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    # scheduler.step()
stop = time.time()

# Evaluate Model
test_logits = genre_clsfr(testdata.X)
test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))
    return acc 

print(f"Training time: {stop - start}s",accuracy_fn(test_pred,testdata.y))
