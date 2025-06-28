import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention,self).__init__()

        # Squeeze operation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
        # Excitation operation
        self.fc1 = nn.Linear(in_channels, in_channels//ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels//ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
    
        excitation = self.fc1(avg_pool)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
    
        attention_map = excitation.view(x.size(0), x.size(1), 1, 1)
    
        return x * attention_map

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size = 7, padding = 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.conv1(x)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map

class face_mask_detection(nn.Module):
    def __init__(self):
        super(face_mask_detection, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
    
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
    
        self.channel_attention1 = ChannelAttention(64,16)
    
        self.spatial_attention1 = SpatialAttention(64)
    
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)
    
        self.channel_attention2 = ChannelAttention(128,16)
    
        self.spatial_attention2 = SpatialAttention(128)
    
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128*13*13, 128)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
    
        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
    
        self.fc3 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.channel_attention1(x)

        x = self.spatial_attention1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
    
        x = self.channel_attention2(x)
    
        x = self.spatial_attention2(x)
    
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout1(x)
    
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.dropout2(x)
    
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x