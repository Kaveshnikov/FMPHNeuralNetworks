import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

dataset = "CIFAR10"

# Download training and testing data from open datasets.
if dataset == 'MNIST':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    training_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
elif dataset == 'SVHN':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_data = datasets.SVHN(root='data', split='train', download=True, transform=transform)
    test_data = datasets.SVHN(root='data', split='test',  download=True, transform=transform)
elif dataset == 'FashionMNIST':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
elif dataset == 'CIFAR10':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
else:
    print('Dataset not recognized!')
    exit(1)

batch_size = 64
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    inp_shape = X.shape
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_shape=(64, 1, 28, 28), num_classes=10):
        super(NeuralNetwork, self).__init__()

        # FIXME: Define the convolution layers (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
        self.features = nn.Sequential(
            # (convolution + act. function + Pooling) x n
        )
        # FIXME: Define the fully connected layers (classifier) with dropout
        self.classifier = nn.Sequential(
            # similarly as on the last exercise
        )

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.classifier.apply(init_weights)
        self.features.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = NeuralNetwork(input_shape=inp_shape).to(device)
print(model)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)


loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# FIXME define a scheduler and set learning rate decay to better train the model in only 5 epochs!
# example: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
scheduler = 42


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")