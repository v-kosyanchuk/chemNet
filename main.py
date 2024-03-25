import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
import time
import sys


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#  ---------------  Joined Model 3 inner layers ---------------
class ChemNetN3(nn.Module):

    def __init__(self, h=100, alpha=0.1, method='LRELU'):
        super().__init__()
        self.fc1 = nn.Linear(12, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, h)
        self.fc5 = nn.Linear(h, 10)

        if method == 'LRELU':
            self.relu = nn.LeakyReLU(alpha)
        if method == 'PRELU':
            self.relu = nn.PReLU(init=0.1)
        if method == 'CELU':
            self.relu = nn.CELU(alpha)
        if method == 'SIGM':
            self.relu = nn.Sigmoid()
        if method == 'TANH':
            self.relu = nn.Tanh()

    def forward(self, x):
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 11]).to(device)
        identity = torch.index_select(x, 1, indices)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = x + identity
        return x.squeeze()


#  ---------------  Joined Model  4 inner layers ---------------
class ChemNetN4(nn.Module):

    def __init__(self, h=100, alpha=0.1, method='LRELU'):
        super().__init__()
        self.fc1 = nn.Linear(12, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, h)
        self.fc5 = nn.Linear(h, h)
        self.fc10 = nn.Linear(h, 10)

        if method == 'LRELU':
            self.relu = nn.LeakyReLU(alpha)
        if method == 'PRELU':
            self.relu = nn.PReLU(init=0.1)
        if method == 'CELU':
            self.relu = nn.CELU(alpha)
        if method == 'SIGM':
            self.relu = nn.Sigmoid()
        if method == 'TANH':
            self.relu = nn.Tanh()

    def forward(self, x):

        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 11]).to(device)
        identity = torch.index_select(x, 1, indices)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc10(x)
        x = x + identity
        return x.squeeze()



@torch.inference_mode()
def calculate_val_loss(_model, _Xv, _Yv, _criterion):
    y_pred = _model(_Xv)
    return torch.sqrt(_criterion(y_pred, _Yv))


def train_model(_model, _criterion, _optimizer, _scheduler, _X, _Y, _Xv, _Yv, _BS, _SH, _n_epochs):
    loss_hist = []  # for plotting
    val_hist = []  # for plotting
    time_hist = []
    tot_time = []

    MM = int(2000000/_BS)

    _X2 = _X
    _Y2 = _Y

    st0 = time.time()

    for epoch in range(_n_epochs):
        hist_loss = 0

        st = time.time()

        if epoch % _SH == 0:
            idx = torch.randperm(_X2.size(0))
            _X2 = _X2[idx, :]
            _Y2 = _Y2[idx]

        for ii in range(0, MM):
            # parse batch
            x = _X2[ii * _BS:ii * _BS + _BS, :]
            y = _Y2[ii * _BS:ii * _BS + _BS, :]

            # sets the gradients of all optimized tensors to zero.
            _optimizer.zero_grad()
            # get outputs
            y_pred = _model(x)
            # calculate loss
            loss = torch.sqrt(_criterion(y_pred, y))
            # calculate gradients
            loss.backward()
            # performs a single optimization step (parameter update)
            _optimizer.step()
            hist_loss += loss.item()
        loss_hist.append(hist_loss / MM)

        val_hist.append(calculate_val_loss(_model, _Xv, _Yv, _criterion))
        _scheduler.step(val_hist[epoch])

        et = time.time()
        time_hist.append(et - st)
        tot_time.append(et - st0)
        print(f"Epoch={epoch} train_loss={loss_hist[epoch]:.6e} val_loss={val_hist[epoch]:.6e}  time:{time_hist[epoch]:.2f}")

        if _optimizer.param_groups[0]['lr'] < 1.01e-6:
            break

    return loss_hist, val_hist, time_hist, tot_time



set_random_seed(42)


# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




DF = pd.read_csv("Data_train_normalized_named.dat", index_col=False, delimiter=' ')
X = torch.tensor(DF[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']].values).to(torch.float32).to(device)
Y = torch.tensor(DF[['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7',  'Y8', 'Y9', 'Y12']].values).to(torch.float32).to(device)

DFv = pd.read_csv("Data_val_normalized_named.dat", index_col=False, delimiter=' ')
Xv = torch.tensor(DFv[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']].values).to(torch.float32).to(device)
Yv = torch.tensor(DFv[['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y12']].values).to(torch.float32).to(device)

DF = pd.read_csv("Data_test_normalized_named.dat", index_col=False, delimiter=' ')
Xt = torch.tensor(DF[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']].values).to(torch.float32).to(device)
Yt = torch.tensor(DF[['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7',  'Y8', 'Y9', 'Y12']].values).to(torch.float32).to(device)



# Loss function
criterion = nn.MSELoss()



original_stdout = sys.stdout


model = ChemNetN3(h=100, alpha=0.10, method='LRELU').to(device)
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=50, threshold=1e-3, cooldown=0, min_lr=1e-6, verbose=True)
loss_history, val_history, time_history, total_time = train_model(model, criterion, optimizer, scheduler, X, Y, Xv,Yv, 2000, 5, 5000)
torch.save(model.state_dict(),"ARTICLE/models/N3_h=100_LRELU.pt")
file1 = open("ARTICLE/metrics/N3_h=100_LRELU.txt", 'w')
for i in range(0, len(loss_history)):
    file1.write(f"{loss_history[i]:.4e} {val_history[i]:.4e} {time_history[i]:.2f} {total_time[i]:.2f} \n")
file1.close()

file2 = open("ARTICLE/output/N3_h=100_LRELU_single.dat", 'w')
fA = model(Xt).detach().cpu().numpy()
sys.stdout = file2
for i in range(0, fA.shape[0]):
    print(*fA[i])
sys.stdout = original_stdout
file2.close()

file3 = open("ARTICLE/output/N3_h=100_LRELU_conseq.dat", 'w')
sys.stdout = file3
for i in range(0, 500):
    x0 = Xt[i * 1000]
    for j in range(0, 1000):
        x1 = model(x0.view(1, 12))
        x0 = torch.tensor([x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], x1[7], x1[8], x0[9], x0[10], x1[9]]).to(device)
        print(*x1.detach().cpu().numpy())
sys.stdout = original_stdout
file3.close()


model = ChemNetN4(h=300, alpha=0.10, method='SIGM').to(device)
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=50, threshold=1e-3, cooldown=0, min_lr=1e-6, verbose=True)
loss_history, val_history, time_history, total_time = train_model(model, criterion, optimizer, scheduler, X, Y, Xv,Yv, 2000, 5, 5000)
torch.save(model.state_dict(),"ARTICLE/models/N4_h=300_SIGM.pt")
file1 = open("ARTICLE/metrics/N4_h=300_SIGM.txt", 'w')
for i in range(0, len(loss_history)):
    file1.write(f"{loss_history[i]:.4e} {val_history[i]:.4e} {time_history[i]:.2f} {total_time[i]:.2f} \n")
file1.close()

file2 = open("ARTICLE/output/N4_h=300_SIGM_single.dat", 'w')
fA = model(Xt).detach().cpu().numpy()
sys.stdout = file2
for i in range(0, fA.shape[0]):
    print(*fA[i])
sys.stdout = original_stdout
file2.close()

file3 = open("ARTICLE/output/N4_h=300_SIGM_conseq.dat", 'w')
sys.stdout = file3
for i in range(0, 500):
    x0 = Xt[i * 1000]
    for j in range(0, 1000):
        x1 = model(x0.view(1, 12))
        x0 = torch.tensor([x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], x1[7], x1[8], x0[9], x0[10], x1[9]]).to(device)
        print(*x1.detach().cpu().numpy())
sys.stdout = original_stdout
file3.close()




model = ChemNetN4(h=300, alpha=0.10, method='CELU').to(device)
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=50, threshold=1e-3, cooldown=0, min_lr=1e-6, verbose=True)
loss_history, val_history, time_history, total_time = train_model(model, criterion, optimizer, scheduler, X, Y, Xv,Yv, 2000, 5, 5000)
torch.save(model.state_dict(),"ARTICLE/models/N4_h=300_CELU.pt")
file1 = open("ARTICLE/metrics/N4_h=300_CELU.txt", 'w')
for i in range(0, len(loss_history)):
    file1.write(f"{loss_history[i]:.4e} {val_history[i]:.4e} {time_history[i]:.2f} {total_time[i]:.2f} \n")
file1.close()

file2 = open("ARTICLE/output/N4_h=300_CELU_single.dat", 'w')
fA = model(Xt).detach().cpu().numpy()
sys.stdout = file2
for i in range(0, fA.shape[0]):
    print(*fA[i])
sys.stdout = original_stdout
file2.close()

file3 = open("ARTICLE/output/N4_h=300_CELU_conseq.dat", 'w')
sys.stdout = file3
for i in range(0, 50):
    x0 = Xt[i * 1000]
    for j in range(0, 1000):
        x1 = model(x0.view(1, 12))
        x0 = torch.tensor([x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], x1[7], x1[8], x0[9], x0[10], x1[9]]).to(device)
        print(*x1.detach().cpu().numpy())
sys.stdout = original_stdout
file3.close()






model = ChemNetN4(h=300, alpha=0.10, method='LRELU').to(device)
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=50, threshold=1e-3, cooldown=0, min_lr=1e-6, verbose=True)
loss_history, val_history, time_history, total_time = train_model(model, criterion, optimizer, scheduler, X, Y, Xv,Yv, 2000, 5, 5000)
torch.save(model.state_dict(),"ARTICLE/models/N4_h=300_LRELU.pt")
file1 = open("ARTICLE/metrics/N4_h=300_LRELU.txt", 'w')
for i in range(0, len(loss_history)):
    file1.write(f"{loss_history[i]:.4e} {val_history[i]:.4e} {time_history[i]:.2f} {total_time[i]:.2f} \n")
file1.close()

file2 = open("ARTICLE/output/N4_h=300_LRELU_single.dat", 'w')
fA = model(Xt).detach().cpu().numpy()
sys.stdout = file2
for i in range(0, fA.shape[0]):
    print(*fA[i])
sys.stdout = original_stdout
file2.close()

file3 = open("ARTICLE/output/N4_h=300_LRELU_conseq.dat", 'w')
sys.stdout = file3
for i in range(0, 50):
    x0 = Xt[i * 1000]
    for j in range(0, 1000):
        x1 = model(x0.view(1, 12))
        x0 = torch.tensor([x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], x1[7], x1[8], x0[9], x0[10], x1[9]]).to(device)
        print(*x1.detach().cpu().numpy())
sys.stdout = original_stdout
file3.close()






