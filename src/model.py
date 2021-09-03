import torch.nn as nn
import torch.nn.functional as F
import torch


class DuelingQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(DuelingQNetwork, self).__init__()

        # value network
        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc4_val = nn.Linear(hidsize2, 1)

        # advantage network
        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc4_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc4_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()

class DuelingConvSimpleQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, node_size, out_channels=5, hidsize1=128, hidsize2=128):
        super(DuelingConvQNetwork, self).__init__()

        self.conv_out_size = int(out_channels * (state_size / node_size))

        
        # value network
        self.fc_conv_val = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=node_size, stride=node_size)
        self.fc1_val = nn.Linear(self.conv_out_size, hidsize1)
        # self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc4_val = nn.Linear(hidsize2, 1)

        # advantage network
        self.fc_conv_adv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=node_size, stride=node_size)
        self.fc1_adv = nn.Linear(self.conv_out_size, hidsize1)
        # self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc4_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        val = F.relu(self.fc_conv_val(x))
        val = val.view(-1, self.conv_out_size)
        val = F.relu(self.fc1_val(val))
        # val = F.relu(self.fc2_val(val))
        val = self.fc4_val(val)

        # advantage calculation
        adv = F.relu(self.fc_conv_adv(x))
        adv = adv.view(-1, self.conv_out_size)
        adv = F.relu(self.fc1_adv(adv))
        # adv = F.relu(self.fc2_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()

class DuelingConvQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, node_size, out_channels=5, hidsize1=128, hidsize2=128):
        super(DuelingConvQNetwork, self).__init__()

        self.conv_out_size = int(out_channels * (state_size / node_size))

        
        # value network
        self.fc_conv_val = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=node_size, stride=node_size)
        self.fc1_val = nn.Linear(self.conv_out_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc4_val = nn.Linear(hidsize2, 1)

        # advantage network
        self.fc_conv_adv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=node_size, stride=node_size)
        self.fc1_adv = nn.Linear(self.conv_out_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc4_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        val = F.relu(self.fc_conv_val(x))
        val = val.view(-1, self.conv_out_size)
        val = F.relu(self.fc1_val(val))
        val = F.relu(self.fc2_val(val))
        val = self.fc4_val(val)

        # advantage calculation
        adv = F.relu(self.fc_conv_adv(x))
        adv = adv.view(-1, self.conv_out_size)
        adv = F.relu(self.fc1_adv(adv))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()