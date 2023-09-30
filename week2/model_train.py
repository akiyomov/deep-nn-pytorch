import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)

model = LinearRegression(1, 1)
criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_data = torch.Tensor([[1], [2], [3], [4], [5]])
y_data = torch.Tensor([[2], [4], [6], [8], [10]])

dataset = TensorDataset(x_data, y_data)

train_loader = DataLoader(dataset=dataset, batch_size=1)

for epoch in range(1000):
    for x, y in train_loader:
        y_pred = model(x)
        loss = criteria(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
