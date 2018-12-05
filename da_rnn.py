"""
Yao Qin. "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
https://arxiv.org/pdf/1704.02971.pdf

"""
import time

import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots

torch.cuda.manual_seed_all(42)
DEVICE = torch.device("cuda")

def gen_data():
    """
    Generate test data
    """
    X = np.random.normal(size=(20000, 1, 24))
    W = np.random.normal(size=(24))
    Y = [1]
    for i in range(X.shape[0]):
        Y.append(0.5 * np.exp(np.dot(X[i].flatten(), W) / 20) + 0.5 * Y[-1])
    Y = np.array(Y[1:])
    X_T = torch.Tensor(X).to(DEVICE)
    Y_T = torch.Tensor(Y).to(DEVICE)
    # plt.plot(Y)
    # plt.show()
    return X_T, Y_T


def maem(pred, fact, mean=True):
    fact = fact.flatten()
    pred = pred.flatten()
    mae = np.mean(np.abs(fact - pred))
    m = np.mean(fact)
    return mae * 100 / m

class RNN(nn.Module):
    """
    LSTMCell implementation
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.RNN = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.RNN = nn.LSTMCell(input_size, hidden_size)
        self.W = nn.Linear(hidden_size, output_size)
        # self.h = torch.zeros((1,hidden_size))
        # self.c = torch.zeros((1,hidden_size))

    def forward(self, x):
        h_t = torch.zeros((1, self.hidden_size)).to(DEVICE)
        c_t = torch.zeros((1, self.hidden_size)).to(DEVICE)
        res = torch.empty(x.shape[0]).to(DEVICE)
        for i in range(x.shape[0]):
            h_t, c_t = self.RNN(x[i], (h_t, c_t))
            res[i] = self.W(h_t)
        # self.c = c_t
        # self.h = h_t
        print()
        return res

X_T, Y_T = gen_data()

net = RNN(X_T.shape[2], X_T.shape[2], 1).to(DEVICE)
# optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0, amsgrad=False)
optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.975, weight_decay=0, nesterov=True)
criterion = nn.MSELoss()

epochs = 100
mb_size = 100
tic = time.perf_counter()
for e in range(epochs):
    for i in range(round(X_T.shape[0]/mb_size)):
        x_mb = X_T[i*mb_size:i*mb_size+mb_size,:,:]
        y_mb = Y_T[i*mb_size:i*mb_size+mb_size]

        pred = net(x_mb)
        loss = criterion(pred, y_mb)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if not(e % int(epochs/10)):
        print(e)
        with torch.no_grad():
            print(float(criterion(net(X_T), Y_T)))
toc = time.perf_counter()
print(DEVICE)
print('Time: ', toc-tic)
with torch.no_grad():
    pred = net(X_T)
    maem_train = maem(pred.cpu().numpy(), Y_T.cpu().numpy())
print("MAEM: ", maem_train)
plt.plot(Y_T.cpu().numpy()[:300])
plt.plot(pred.cpu().numpy()[:300])
plt.show()
# tsaplots.plot_acf(Y, lags=10)
# tsaplots.plot_pacf(Y, lags=10)
# plt.show()
