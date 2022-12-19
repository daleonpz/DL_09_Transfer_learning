import torch


class MiniModel(torch.nn.Module):
    def __init__(self, in_channel):
        super(MiniModel, self).__init__()

        self.linear1 = torch.nn.Linear(in_channel, 128)
        self.linear2 = torch.nn.Linear(128, 8)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.activation(x)
        return x
