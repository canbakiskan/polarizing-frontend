"""
Author: Can Bakiskan
Date: 2019-09-03

"""
import torch
from torch import nn
import torch.nn.functional as F
from normalized_conv2d import Normalized_Conv2d, Saturation_activation


class Classifier(nn.Module):
    def __init__(self, in_channels=1, **kwargs):

        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Classifier_no_maxpool(nn.Module):
    def __init__(self, in_channels=1, **kwargs):

        super(Classifier_no_maxpool, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.fc1 = nn.Linear(15488, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 15488)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Direct_quantization_model(nn.Module):
    def __init__(self, in_channels=1, jump=0.2, bpda_steepness=16, **kwargs):

        super(Direct_quantization_model, self).__init__()
        # self.frontend = Normalized_Conv2d(
        #     in_channels=in_channels,
        #     out_channels=1,
        #     kernel_size=5,
        #     jump=jump,
        #     bpda_steepness=bpda_steepness,
        #     padding=2,
        # )
        # self.frontend.weight.data[0, 0, :, :] = 0.0
        # self.frontend.weight.data[0, 0, 3, 3] = 1.0
        # self.frontend.weight.requires_grad = False
        self.jump = nn.Parameter(torch.tensor(jump, dtype=torch.float))
        self.bpda_steepness = nn.Parameter(torch.tensor(jump, dtype=torch.float))
        self.frontend = Saturation_activation().apply

        self.classifier = Classifier(in_channels=1)

    def forward(self, x):

        x = self.frontend(x, self.jump, self.bpda_steepness)
        x = self.classifier(x)

        return x


class Polarization_quantization_model(nn.Module):
    def __init__(self, in_channels=1, jump=0.2, bpda_steepness=16, **kwargs):

        super(Polarization_quantization_model, self).__init__()

        self.frontend = Normalized_Conv2d(
            in_channels=in_channels,
            out_channels=25,
            kernel_size=5,
            jump=jump,
            bpda_steepness=bpda_steepness,
            padding=2,
        )

        nn.init.kaiming_uniform_(self.frontend.weight, nonlinearity="relu")
        self.frontend.weight.requires_grad = True

        self.classifier = Classifier(in_channels=self.frontend.out_channels)

    def forward(self, x):

        x = self.frontend(x)
        x = self.classifier(x)

        return x

    def set_bpda_steepness(self, bpda_steepness):
        self.frontend.set_bpda_steepness(bpda_steepness)


class Polarization_quantization_model_no_maxpool(nn.Module):
    def __init__(self, in_channels=1, jump=0.2, bpda_steepness=16, **kwargs):

        super(Polarization_quantization_model_no_maxpool, self).__init__()

        self.frontend = Normalized_Conv2d(
            in_channels=in_channels,
            out_channels=25,
            kernel_size=5,
            jump=jump,
            bpda_steepness=bpda_steepness,
            padding=2,
        )

        nn.init.kaiming_uniform_(self.frontend.weight, nonlinearity="relu")
        self.frontend.weight.requires_grad = True

        self.classifier = Classifier_no_maxpool(in_channels=self.frontend.out_channels)

    def forward(self, x):

        x = self.frontend(x)
        x = self.classifier(x)

        return x

    def set_bpda_steepness(self, bpda_steepness):
        self.frontend.set_bpda_steepness(bpda_steepness)


class Blackbox_model(nn.Module):
    def __init__(self):

        super(Blackbox_model, self).__init__()
        # self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4)

        self.dense1 = nn.Linear(in_features=1024, out_features=256)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(in_features=256, out_features=64)

        self.dense3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):

        # x = self.bn1(x)

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)  # reshape
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        return x
