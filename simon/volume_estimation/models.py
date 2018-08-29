import torch
import torch.nn as nn

class BasicConvnet(nn.Module):
    """
    Simple small convnet which output an estimated volume of the fish given
    the filtered and normalized depth map 
    
    Input:
        - dmap : torch tensor of size (N, W, H)
    Output:
        - volume : torch tensor of size (1, )
    """
    def __init__(self, target_size):
        super(BasicConvnet, self).__init__()
        self.target_size = target_size
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16,
                                             kernel_size=5, stride=1, padding=2), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32,
                                             kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64,
                                             kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.dense = nn.Linear(28416, target_size)
        self.is_training = True
    
    def forward(self, dmap):
        output_1 = self.conv1(dmap)
        output_2 = self.conv2(output_1)
        output_3 = self.conv3(output_2)
        flattened_output = output_3.view(output_3.size(0), -1)
        volume = self.dense(flattened_output)
        
        return volume.squeeze(1)
    
class MultiHeadConvnet(nn.Module):
    """
    Simple small convnet with two regression head.
    The first head purpose is to predict spatial labels : 'height', 'width'
    & 'length'
    The second head is a feed forward net which learns how to recombine 
    estimators of 'height', 'width' & 'length' to predict the volume.
    
    Input:
        - dmap : torch tensor of size (N, W, H)
    Output:
        - spatial_extents: torch tensor of size (N, 3)
        - volume: torch tensor of size(N, 1)
    """
    def __init__(self):
        super(MultiHeadConvnet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16,
                                             kernel_size=5, stride=1, padding=2), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32,
                                             kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.spatial_dense = nn.Linear(60000, 3)
        self.non_linearity = nn.ReLU()
        self.volume_dense = nn.Linear(3, 1)
        self.dropout = nn.Dropout2d(0.2)
        self.is_training = True
        
    def forward(self, dmap):
        output_1 = self.conv1(dmap)
        output_2 = self.conv2(output_1)
        flattened_output = output_2.view(output_2.size(0), -1)
        spatial_extents = self.spatial_dense(flattened_output)
        dropout_spatial = self.dropout(self.non_linearity(spatial_extents))
        volume = self.volume_dense(self.non_linearity(spatial_extents))
        
        return spatial_extents.squeeze(1), volume.squeeze(1)