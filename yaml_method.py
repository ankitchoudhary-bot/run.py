import torch
import argparse
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--threshold", type=float, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loader
def MNIST_loaders(train_batch_size=5000, test_batch_size=1000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(MNIST('./data/', train=True, download=True, transform=transform), batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(MNIST('./data/', train=False, download=True, transform=transform), batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader

def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

class Layer(nn.Linear):
    def __init__(self, in_features, out_features, lr, threshold, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=lr)
        self.threshold = threshold
        self.num_epochs = 2

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for _ in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

class Net(torch.nn.Module):
    def __init__(self, dims, lr, threshold):
        super().__init__()
        self.layers = [Layer(dims[i], dims[i+1], lr, threshold).to(device) for i in range(len(dims)-1)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(h.pow(2).mean(1))
            goodness_per_label.append(sum(goodness).unsqueeze(1))
        return torch.cat(goodness_per_label, 1).argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for layer in self.layers:
            h_pos, h_neg = layer.train(h_pos, h_neg)

# Main training & evaluation block (now inside the function)
torch.manual_seed(1234)
train_loader, test_loader = MNIST_loaders()
net = Net([784, 50, 50], args.lr,args.threshold)
x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)
x_pos = overlay_y_on_x(x, y)
rnd = torch.randperm(x.size(0))
x_neg = overlay_y_on_x(x, y[rnd])
net.train(x_pos, x_neg)

x_te, y_te = next(iter(test_loader))
x_te, y_te = x_te.to(device), y_te.to(device)
acc = net.predict(x_te).eq(y_te).float().mean().item()
print(f"accuracy={acc}")
