
from sklearn.utils.class_weight import compute_class_weight
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
from model import AdvancedNeuralNet
from nltk_utils import tokenize
import re
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

'''
Again thanks to Yin for finding the article on this
Link to the article: https://medium.com/@shrinidhi.rm1990/simple-chatbot-using-bert-and-pytorch-part-1-2735643e0baa
'''
# Covert json data to a dataframe
rows = []
for intent in intents['intents']:
    # Flatten the patterns
    for pattern in intent['patterns']:
        pattern = re.sub(r'[^a-zA-Z ]+', '', pattern) #standard regex for letters only
        rows.append([pattern, intent['tag']])

filename = "intents.csv"
# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['text', 'label'])
    csvwriter.writerows(rows)

df = pd.read_csv(filename)

# Converting the labels into encodings
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
train_text, train_labels = df['text'], df['label']

# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

max_seq_len = 8

# tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_text.tolist(),
    max_length=max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# define a batch size
batch_size = 16
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# DataLoader for train set
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size)

# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
    param.requires_grad = False

# Hyperparameters

output_size = len(tags)
learning_rate = 0.001
input_size =  500
hidden_size = 200 
# number of training epochs
epochs = 450

# specify GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedNeuralNet(bert, input_size, hidden_size, output_size)
# push the model to GPU
model = model.to(device)
print(summary(model))

# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# compute the class weights
class_wts = compute_class_weight(
    class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
print(f"class weights: {class_wts}")

# convert class weights to tensor
weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.to(device)
# loss function
cross_entropy = nn.NLLLoss(weight=weights)

# empty lists to store training and validation loss of each epoch
train_losses = []

# function to train the model


def train():

    model.train()
    total_loss = 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(
                step,    len(train_dataloader)))
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the    exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # clear calculated gradients
        optimizer.zero_grad()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds)
        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)

        # predictions are in the form of (no. of batches, size of batch, no. of classes).
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = np.concatenate(total_preds, axis=0)
        # returns the loss and predictions
        return avg_loss, total_preds


for epoch in range(epochs):
    # train model
    train_loss, _ = train()

    # append training and validation loss
    train_losses.append(train_loss)
    # it can make your experiment reproducible, similar to set  random seed to all options where there needs a random seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if (epoch + 1) % 50 == 0:
        print(f'epoch {epoch+1}/{epochs}, loss={train_loss:.6f}')

"""
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

print(all_words)
ignore_words = ['?', '!', '.', ',']
#We don't want punctuation marks
all_words = [stem(w) for w in all_words if w not in ignore_words]

print("---------------")
print("All our words after tokenization")
print(all_words)

all_words = sorted(set(all_words))
tags = sorted(set(tags))

#Now we are creating the lists to train our data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

#Convert into a numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)


#Create a new Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#How do these hyperparameters affect optimization of our chatbot?:
batch_size = 8 #batch size for batch training to make loading faster -> number of parts used in one iteration
hidden_size = 8 #number of features of a hidden state for the nn
output_size = len(tags) #the number of different classes (tags) we are outputting to
learning_rate = 0.001 #step size at each iteration while moving towards a (local OR global) minimum for our loss function
num_epochs = 1000 #number of times the training process will repeat

input_size = len(X_train[0])
print("Below is the Input Size of our Neural Network")
print(input_size, len(all_words))
print("Below is the output size of our neural network, which should match the amount of tags ")
print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#The below function helps push to GPU for training if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
#model = AdvancedNeuralNet(bert, input_size, hidden_size, output_size)

#Loss and Optimizer

#Experiment with another optimizer and note any differences in loss of our model. Does the final loss increase or decrease? 
#Final loss changed from 0.0003 to 1.7095 using the SGD optimizer

#Speculate on why your changed optimizer may increase or decrease final loss
#SGD is regarded as being pretty unstable, being noisy and not performing well with small datasets like ours, adam is a lot more efficient in that regard
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #originally tried doing this with LBFGS but threw an error, I can't be bothered

for epoch in range(num_epochs):
    correct = 0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(words)

        loss = criterion(outputs, labels.type(torch.LongTensor)) #need to cast to longtensor here to not throw an error

        #backward and optimizer step 
        optimizer.zero_grad()

        #Calculate the backpropagation
        loss.backward()
        optimizer.step()

    #Print progress of epochs and loss for every 100 epochs
    if (epoch +1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}, accuracy={accuracy:.4f}')

print(f'final loss, loss={loss.item():.4f}')

    #Need to save the data 
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}
"""

# Need to save the data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "bert": bert,
    "all_words": all_words,
    "tags": tags,
    "tokenizer": tokenizer,
    "max_seq_len": max_seq_len,
    'le': le,
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete, file saved to {FILE}')
#Should save our training data to a pytorch file called "data"