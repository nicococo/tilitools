import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import transforms

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda else "cpu")


class SimpleMnistAutoencoder(nn.Module):
    latent = torch.Tensor

    def __init__(self, hidden):
        super(SimpleMnistAutoencoder, self).__init__()
        self.fc1 = nn.Linear(784, 400, bias=False)
        self.fc2 = nn.Linear(400, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, 400, bias=False)
        self.fc4 = nn.Linear(400, 784, bias=False)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        # store the latent variables
        self.latent = x
        x = self.ReLU(self.fc3(x))
        x = self.ReLU(self.fc4(x))
        return x


transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_loader = torch.utils.data.DataLoader(dataset=mnist_trainset, batch_size=1, shuffle=False)
print(mnist_trainset)

x = torch.zeros((100, 784))
idx = 0
for bidx, (input, target) in enumerate(mnist_loader):
    if target == 2:
        x[idx, :] = input.view(784)
        idx += 1
        if idx >= x.size()[0]:
            break



# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H = 100, 784, 50

model = SimpleMnistAutoencoder(H)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(size_average=False)

model.train()   # some layers (batchnorm) behave differently in training and testing.
learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, x)
    print(t, '{0:1.6f}'.format(loss.item()))

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()
    optimizer.step()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    # with torch.no_grad():
    #     for param in model.parameters():
    #         param -= learning_rate * param.grad
print(model.latent)
print(model.latent.size())


plt.figure(1)
plt.subplot(2, 1, 1)
plt.imshow(y_pred[0, :].view(28, 28).detach().numpy())
plt.subplot(2, 1, 2)
plt.imshow(x[0, :].view(28, 28).detach().numpy())
plt.show()

model.eval()  # some layers (batchnorm) behave differently in training and testing.