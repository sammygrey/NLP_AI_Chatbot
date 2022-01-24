import json
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words
#Reading in our json file and loading as intents, changes it into a python object
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
#Essentially look into the "intents" key in the json file, then the "tag" key under intents
# We append our tag key onto our tag array
# For each pattern in our intents with the key "patterns", we apply a tokenization
# We then place it into the all words array. This is essentially all our words in the intents.json file
# We extend into the all_words array because we dont want to append arrays onto arrays
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

print(all_words)
ignore_words = ['?', '!', '.', ',']
#We dont want punctuation marks
all_words = [stem(w) for w in all_words if w not in ignore_words]

print("---------------")
print("All our words after stemming")
print(all_words)

#Remove duplicate elements and return a list of our words
# We have now tokenized, stemmed, and excluded punctuation characters from our 
all_words = sorted(set(all_words))
tags = sorted(set(tags))


#Now we are creating the lists to train our data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    #pattern_sentenced is already tokenized
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    #For Y data, we have numerical labels for our tags
    # I.E. Our first tag is indexed as 0, second tag indexed as 1, so forth
    label = tags.index(tag)
    y_train.append(label)
    #Usually, you want a one-hot encoding, but since we are using PyTorch, we are using only class labels

#Convert into a numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)


#Create a new Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #To later access dataset with index idx
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#Hyperparamters
batch_size = 8
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

#Lengths of each bag of words we created, which is the same length of all words
input_size = len(X_train[0])
print("Below is the Input Size of our Neural Network")
print(input_size, len(all_words))
print("Below is the output size of our neural network, which should match the amount of tags ")
print(output_size, tags)

#num_workers is for threading, can set to 0 
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#The below function helps push to GPU for training if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(words)
        #Calculation of the loss is our CrossEntropyLoss with predicted outputs and actual labels
        loss = criterion(outputs, labels)

        #backward and optimizer step 
        optimizer.zero_grad()
        #Calculate the backpropogation
        loss.backward()
        optimizer.step()

    #Print progress of epochs and loss for every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

    print(f'final loss, loss={loss.item():.4f}')