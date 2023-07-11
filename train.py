import json
from nltk_utils import token, stem, bag_of_words
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f: # read mode
    intents = json.load(f)

all_words= []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = token(pattern)
        all_words.extend(w) # adds elements from one array onto another 
        # so [a, b].extend([c, d]) will be [a,b,c,d] 
        # as opposed to append which is [a,b, [c,d]]
        xy.append((w, tag))

ignore_words = ["?", "!", ".", ","]
all_words = [stem(i) for i in all_words if i not in ignore_words]
all_words = sorted(set(all_words)) # unique words only
tags = sorted(set(tags)) 

X_train = []
y_train = []

for (pattern_sen, tag) in xy:
    bag = bag_of_words(pattern_sen, all_words)
    X_train.append(bag)

    label = tags.index(tag) # indexing tags
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.X_data = X_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx] # returned as a tuple
    
    def __len__(self):
        return self.n_samples

# hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)  # or len(X_train[0]) i.e. of first bag of words
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# device = torch.devcice('cuda' if torch.cuda.is_available() else'cpu')
model = NeuralNet(input_size, hidden_size, output_size)#.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        # words = words.to(device)
        # labels = labels.to(device)
        words = words
        labels = labels.to(torch.long)


        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # backward and optimizer
        optim.zero_grad()
        loss.backward()
        optim.step()

    if (epoch+1)%100 == 0:
        print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

print(f"final loss={loss.item():.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags  
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"training complete, file saved to {FILE}")


