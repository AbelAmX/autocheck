import torch
import torch.nn as nn
in_channels = 5
out_channels = 20
kernel_size = 3
m_size = 8
Layer1 = 100
Layer2 = 40


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input channels = 5, output channels =
        self.conv1 = nn.Conv2d(5, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.p_size = out_channels*int(m_size/2)*int(m_size/4)
        self.fc1 = nn.Linear(self.p_size, Layer1)
        self.fc2 = nn.Linear(Layer1, Layer2)
        self.fc = nn.Linear(Layer1, 1)

    def forward(self, x):
        # x = torch.FloatTensor(x)
        x = x.view(-1, 5, m_size, int(m_size/2))
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.p_size)
        x = nn.functional.relu(self.fc1(x))
        value = self.fc(x)
        policy = nn.functional.softmax(self.fc2(x), dim=1)

        return policy, value


class ErrorFnc(nn.Module):
    def __init__(self):
        super(ErrorFnc, self).__init__()

    def forward(self, value, value_estimate, policy, policy_estimate):
        value_error = (value - value_estimate)**2  # mean squared error
        policy_error = torch.sum((policy*(1e-8 + policy_estimate.float()).log()), 1)
        error = (value_error.view(-1).float() + policy_error).mean()
        return error


# inp = torch.randn(1, 5, 6, 6)
# q = nn.conv2d(5, 10, 3, stride=1, padding=1)
# print(inp)
# print(q(inp))
