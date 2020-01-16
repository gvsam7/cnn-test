"""
Author: Georgios Voulgaris
Date: 14/01/2020
Description: Load microsoft kaggle cat/dog dataset, preprocess the images and create and run
convolutional neural networks.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


# Data loading and processing
REBUILD_DATA = True

class DogsVSCats():
    IMG_SIZE = 50  # normalises images on 50 x 50 sizes
    CATS = "PetImages/CAT"
    DOGS = "PetImages/DOGS"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0
    dogcount = 1

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.lisdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.RESIZE(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data_append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ", self.dogcount)


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))
# print(training_data[1])

# plt.imshow(training_data[1][0], cmap="gray")
# plt.show()

# Create a CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        """
        In order to pass data on the dense layer (linear in pytorch), data needs to be flattened first. 
        Below is the formula that achieves this:
        initial image size = 50
        kernel k = 5
        Stride s = 1
        Padding P = 0
        
        The formula for the number of outputs to the next layer of conv2d is: 
        O = { (W - k + 2 * P) / s } + 1
        So, number of pixels/features after first conv2d layer:
          O1 = {(50 - 5 + 2 * 0) / 1} + 1 = 46
        Next, we apply a maxpool of size (2,2) hence reducing the size by half across both 
        dimensions of image.
        So o/p size after 1st maxpool layer = 46 / 2 = 23
        Next, we apply another conv2d layer with same values. So size of output layer via similar 
        calculations:
          O2 = { (23 - 5 + 2 * 0) / 1} + 1 = 19
          O2_maxPooling = 19 / 2 = 9
          O3 = ((9 - 5 + 2 * 0) / 1) + 1 = 5
          O3_maxPooling = 5 / 2 = 2
          By calculated, the output is 128 * 2 * 2 = 512
        
        self.fc1 = nn.Linear(2 * 2 * 128, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 is the No of classes
        Below a more general flattened approach is followed.
        """

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 is the number of classes

    def convs(self, x):
        # max pooling over 2 x 2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # output layer
        return F.softmax(x, dim=1)


net = Net()

# training loop
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X / 255.0  # scaling img so pixels be between 0 - 1
y = torch.Tensor([i[1] for i in training_data])

# separate data for validation
VAL_PCT = 0.1  # lets reserve 10% of the data for validation
val_size = int(len(X) * VAL_PCT)  # convert to int because this No will be used to slice the data into groups
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X), len(test_X))

# iterate over this data to fit and test
BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        #print(f"{i}: {i:i+BATCH_SIZE}
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()  # performs the update

    print(f"Epoch: {epoch}. Loss: {loss}")

# Validation
correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list
        predicted_class = torch.argmax(net_out)

        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))

