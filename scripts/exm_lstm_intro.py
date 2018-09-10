import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim

from tilitools.utils_data import get_2state_anom_seq


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=False)
        self.relu = nn.ReLU()

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform(self.lstm.weight_hh_l0)

    def forward(self, input):
        # tt = torch.cuda if self.isCuda else torch
        # h0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        # c0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        encoded_input, hidden = self.lstm(input)
        # encoded_input = self.relu(encoded_input)
        return encoded_input


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, bias=True, batch_first=False)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform(self.lstm.weight_hh_l0)

    def forward(self, encoded_input):
        # tt = torch.cuda if self.isCuda else torch
        # h0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        # c0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        decoded_output, hidden = self.lstm(encoded_input)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, isCuda)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output


N = 100
seq_len = 20
features = 1

x = torch.randn(N, seq_len, features)
y = torch.randn(N, seq_len, features)

for i in range(N):
    a, b, _ = get_2state_anom_seq(seq_len, 8, anom_prob=1.0, num_blocks=1)
    x[i, :, :] = torch.tensor(a.T)
    y[i, :, :] = torch.tensor(b.T)

# model = torch.nn.LSTM(input_size=features,
#                       hidden_size=1, num_layers=3, bias=True, bidirectional=False, batch_first=False)


model = LSTMAE(features, 20, 2, False)

for i in range(100):
    output = model(x[:, :, :])
    loss_fn = torch.nn.MSELoss(size_average=False)
    my_loss = loss_fn(output, y)
    print(my_loss)

    model.zero_grad()
    my_loss.backward()
    learning_rate = 1e-4
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad


print(output[1, :, :], y[1, :, :])