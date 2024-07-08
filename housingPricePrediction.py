import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
'''
Format data (convert boolean values and furnishing status to 0's, 1's, 2's, etc)
Split dataset into x_train, y_train, x_test, y_test where the y values are the housing prices
make a model and scale values appropriately
make a training and test loop
check and plot loss, accuracy, etc to make sure everything is working properly
sample from the dataset, input a value and see is price matches up

potentially add batch normalization for practice
'''

housingPrice_df = pd.read_csv('housingPriceDataset.csv')
housingPrice_df = housingPrice_df.sample(frac=1)

#changing everything to numerical values
housingPrice_df = housingPrice_df.replace('yes', 1)
housingPrice_df = housingPrice_df.replace('no', 0)
housingPrice_df = housingPrice_df.replace('unfurnished', 0)
housingPrice_df = housingPrice_df.replace('semi-furnished', 1)
housingPrice_df = housingPrice_df.replace('furnished', 2)

#house prices
price_df = housingPrice_df.pop('price')


#print(price_df.head())
#print(housingPrice_df.head())

split = int(len(housingPrice_df) * 0.9)

#make training and test sets
x_train, x_test = np.array(housingPrice_df[:split]), np.array(housingPrice_df[split:])
y_train, y_test = np.array(price_df[:split]), np.array(price_df[split:])

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#convert data to torch tensors and scale them
scaler = StandardScaler()
x_train = torch.from_numpy(scaler.fit_transform(x_train)).type(torch.float32)
x_test = torch.from_numpy(scaler.transform(x_test)).type(torch.float32)
y_train = torch.from_numpy(scaler.fit_transform(y_train)).type(torch.float32)
y_test = torch.from_numpy(scaler.transform(y_test)).type(torch.float32)


'''
print(x_train, len(x_train))
print(x_test, len(x_test))
print(y_train, len(y_train))
print(y_test, len(y_test))
'''

class housePrice(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.random_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
      return self.random_stack(x)
  
model_1 = housePrice(x_train.shape[1])
#print(list(model_1.parameters()))
optimizer = optim.Adam(params=model_1.parameters(), lr=0.00001)
loss_fn = nn.L1Loss()

def train_step(model):
    model.train()
    y_preds = model(x_train)

    loss = loss_fn(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
def test_step(model):
    model.eval()
    with torch.inference_mode():
        y_preds_test = model(x_test)
        loss = loss_fn(y_preds_test, y_test)
    return loss

epochs = 10000
train_losses=[]
test_losses=[]

for epoch in range(epochs):
    train_loss = train_step(model_1)
    test_loss = test_step(model_1)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if epoch % 1000 == 0:
        print(f'Epoch: {epoch} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}')


train_losses_cat = torch.cat([t.unsqueeze(0) for t in train_losses], dim=0).detach().numpy()
test_losses_cat = torch.cat([t.unsqueeze(0) for t in test_losses], dim=0).detach().numpy()



plt.figure(figsize=(8,6))
plt.plot(train_losses_cat, label='Train loss')
plt.plot(test_losses_cat, label='Test loss')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

model_1.eval()
with torch.inference_mode():
    y_pred = model_1(x_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test_original, y_pred)

print(f'R-squared error: {r2:.5f}')


plt.figure(figsize=(8,6))
plt.plot(y_pred, label='Predicted Prices')
plt.plot(y_test_original, label='Actual Prices')
plt.legend()
plt.ylabel('Price')
plt.show()

