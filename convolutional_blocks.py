import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvProcessingBlockResnet(nn.Module):
    '''Implements a basic cascade of 2 convolutional layers, 
    each followed by a Leaky ReLU activation function. 
    Used as the basic building block of the model 
    (repeated num_blocks_per_stage times)'''
    
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(ConvProcessingBlockResnet, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)
        
        # Resnet update
        out += x
        out = F.leaky_relu(out)
        
        print(out.shape)
        return out

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)
        
        # Resnet update
        out += x
        out = F.leaky_relu(out)

        return out


class ConvDimensionalityReductionBlockResnet(nn.Module):
    '''Implements a basic cascade of 2 convolutional layers, 
    with an average pooling layer in the middle, which 
    effectively halves the height and width dimensions of 
    the tensor volume passing through it. Used only as a 
    dimensionality reduction layer, usually once after each 
    network stage. A network stage is considered a cascade 
    of convolutional layers, followed by a dimensionality 
    reduction function such as average pooling.'''
    
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation, reduction_factor):
        super(ConvDimensionalityReductionBlockResnet, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
        self.reduction_factor = reduction_factor
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        out = F.avg_pool2d(out, self.reduction_factor)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        # Downsampling the input so we can add it to the output
        downsample = nn.Conv2d(x.shape[1], out.shape[1], kernel_size=1, stride=2)
        downsampled_x = downsample(x)
        
        out += downsampled_x
        out = F.leaky_relu(out) 
        
        print(out.shape)

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        out = F.avg_pool2d(out, self.reduction_factor)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        # Downsampling the input so we can add it to the output
        downsample = nn.Conv2d(x.shape[1], out.shape[1], kernel_size=1, stride=2)
        downsampled_x = downsample(x)
        
        out += downsampled_x
        out = F.leaky_relu(out) 
        
        return out
    
class ConvProcessingBlockBNNetwork(nn.Module):
    '''Implements a basic cascade of 2 convolutional layers, 
    each followed by a Leaky ReLU activation function. 
    Used as the basic building block of the model 
    (repeated num_blocks_per_stage times)'''
    
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(ConvProcessingBlockBNNetwork, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        return out


class ConvDimensionalityReductionBlockBNNetwork(nn.Module):
    '''Implements a basic cascade of 2 convolutional layers, 
    with an average pooling layer in the middle, which 
    effectively halves the height and width dimensions of 
    the tensor volume passing through it. Used only as a 
    dimensionality reduction layer, usually once after each 
    network stage. A network stage is considered a cascade 
    of convolutional layers, followed by a dimensionality 
    reduction function such as average pooling.'''
    
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation, reduction_factor):
        super(ConvDimensionalityReductionBlockBNNetwork, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
        self.reduction_factor = reduction_factor
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        out = F.avg_pool2d(out, self.reduction_factor)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        out = F.avg_pool2d(out, self.reduction_factor)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        return out