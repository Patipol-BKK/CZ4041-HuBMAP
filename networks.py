import numpy as np
import copy
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, layer_type, in_channels, out_channels, hidden_channels=1, num_layers=1, activation_type='ReLU', negative_slope=0.01, kernel_size=3, padding=1, stride=1):
        super().__init__()
        layers = []

        for _ in range(num_layers):
            if not _:
                layer_in_channels = in_channels
            else:
                layer_in_channels = hidden_channels
                
            if _ < num_layers - 1:
                layer_out_channels = hidden_channels
            else:
                layer_out_channels = out_channels
                
            match layer_type:
                case 'Conv2d':
                    layer = nn.Conv2d(
                        layer_in_channels, 
                        layer_out_channels, 
                        kernel_size = kernel_size,
                        padding = padding,
                        stride = stride
                    )
                case 'ConvTranspose2d':
                    layer = nn.ConvTranspose2d(
                        layer_in_channels, 
                        layer_out_channels, 
                        kernel_size = kernel_size,
                        padding = padding,
                        stride = stride
                    )
                case 'Linear':
                    layer = nn.Linear(
                        layer_in_channels, 
                        layer_out_channels
                    )
            match activation_type:
                case 'ReLU':
                    activation = nn.ReLU()
                case 'LeakyReLU':
                    activation = nn.LeakyReLU(negative_slope = negative_slope)
                case 'Sigmoid':
                    activation = nn.Sigmoid()
                case 'None':
                    activation = None
            layers.append(layer)
            layers.append(nn.BatchNorm2d(layer_out_channels))
            if activation != None:
                layers.append(activation)
        self.sequential = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.sequential.forward(x)

class RecurrentBlock(nn.Module):
    def __init__(self, layer_type, in_channels, out_channels, activation_type, negative_slope=0.01, kernel_size=3):
        super().__init__()
        layers = []
        match layer_type:
            case 'Conv2d':
                layer = nn.Conv2d(
                    out_channels, 
                    out_channels, 
                    kernel_size = kernel_size,
                    padding = 1
                )
            case 'ConvTranspose2d':
                layer = nn.ConvTranspose2d(
                    out_channels, 
                    out_channels, 
                    kernel_size = kernel_size,
                    padding = 1
                )
            case 'Linear':
                layer = nn.Linear(
                    out_channels, 
                    out_channels
                )
        match activation_type:
            case 'ReLU':
                activation = nn.ReLU()
            case 'LeakyReLU':
                activation = nn.LeakyReLU(negative_slope = negative_slope)
            case 'Sigmoid':
                activation = nn.Sigmoid()
            case 'None':
                activation = None
                
        layers.append(layer)
        layers.append(nn.BatchNorm2d(out_channels))
        if activation != None:
            layers.append(activation)
        
        self.sequential = nn.Sequential(*layers)
    
    def forward(self, x):
        x1 = self.sequential(x)
        out = self.sequential(x + x1)
        return out

class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        
        channel_attention = self.sequential(avg_pool.view(x.size(0), -1)) + self.sequential(max_pool.view(x.size(0), -1))
        scale = nn.functional.sigmoid(channel_attention).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x * scale
    
class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        
        self.conv = Block(
            layer_type = 'Conv2d', 
            in_channels = 2, 
            out_channels = 1,
            num_layers = 1, 
            activation_type = 'None', 
            kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2,
            stride = 1
        )
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.conv(x_compress)
        scale = nn.functional.sigmoid(x_out)
        return x * scale
        
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_gate = ChannelGate(in_channels, reduction)
        self.spatial_gate = SpatialGate()
    def forward(self, x):
        flatten_x = x.view(x.size(0), -1)
        channel_out = self.channel_gate(x)
        spatial_out = self.spatial_gate(x)        
        attention = channel_out + spatial_out

        return attention

# Recurrent residual blocks
class CBAM_R2Block(nn.Module):
    def __init__(self, layer_type, in_channels, out_channels, hidden_channels=1, num_layers=1, activation_type='ReLU', negative_slope=0.01, kernel_size=3):
        super().__init__()
        layers = []
        
        for _ in range(num_layers):
            if not _:
                layer_in_channels = in_channels
            else:
                layer_in_channels = hidden_channels
                
            if _ < num_layers - 1:
                layer_out_channels = hidden_channels
            else:
                layer_out_channels = out_channels
                
            layers.append(RecurrentBlock(
                layer_type = layer_type,
                in_channels = layer_in_channels,
                out_channels = layer_out_channels,
                activation_type = activation_type,
                negative_slope = negative_slope,
                kernel_size = kernel_size
            ))
        
        self.sequential = nn.Sequential(*layers)
        
        self.conv = nn.Conv2d(in_channels,hidden_channels,kernel_size=1,stride=1,padding=0)
        
        self.cbam = CBAM(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        r2_out = self.sequential(x) + x
        out = r2_out * self.cbam(r2_out)
        return out

# Recurrent residual blocks
class R2Block(nn.Module):
    def __init__(self, layer_type, in_channels, out_channels, hidden_channels=1, num_layers=1, activation_type='ReLU', negative_slope=0.01, kernel_size=3):
        super().__init__()
        layers = []
        
        for _ in range(num_layers):  
            layers.append(RecurrentBlock(
                layer_type = layer_type,
                in_channels = hidden_channels,
                out_channels = hidden_channels,
                activation_type = activation_type,
                negative_slope = negative_slope,
                kernel_size = kernel_size
            ))
        
        self.sequential = nn.Sequential(*layers)
        
        self.conv_in = nn.Conv2d(in_channels,hidden_channels,kernel_size=1,stride=1,padding=0)
        
        self.conv_out = nn.Conv2d(hidden_channels,out_channels,kernel_size=1,stride=1,padding=0)
        
    def forward(self, x):
        x = self.conv_in(x)
        r2_out = self.sequential(x) + x
        out = self.conv_out(r2_out)
        return out

# Plain Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
#         resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
#         encoder_layers = list(resnet.children())[:-1]
#         self.resnet50_encoder = torch.nn.Sequential(*encoder_layers)
        
#         for param in self.resnet50_encoder.parameters():
#             param.requires_grad = False
        
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
#             nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
            nn.ReLU(),
        )
        
    def forward(self, x):
#         out = self.resnet50_encoder.forward(x)
#         out = torch.reshape(out, (out.size(0), 512, 2, 2))
#         out = self.decoder.forward(out)
        out = self.sequential(x)
        return out

# UNet based on https://bmcbiomedeng.biomedcentral.com/articles/10.1186/s42490-021-00050-y
class UNet(nn.Module):
    def __init__(self, input_size=(128, 128), output_size=(128, 128)):
        super().__init__()
        
        self.transform_input_1 = nn.AdaptiveAvgPool2d(input_size)
        self.transform_output = nn.AdaptiveAvgPool2d(output_size)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.encode_1 = Block(
            layer_type = 'Conv2d',
            in_channels = 1, 
            out_channels = 16,
            hidden_channels = 16,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_2 = Block(
            layer_type = 'Conv2d',
            in_channels = 16, 
            out_channels = 32,
            hidden_channels = 32,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_3 = Block(
            layer_type = 'Conv2d',
            in_channels = 32, 
            out_channels = 64, 
            hidden_channels = 64,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_4 = Block(
            layer_type = 'Conv2d',
            in_channels = 64, 
            out_channels = 128, 
            hidden_channels = 128,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_5 = Block(
            layer_type = 'Conv2d',
            in_channels = 128, 
            out_channels = 128, 
            hidden_channels = 256 ,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_1 = Block(
            layer_type = 'ConvTrans2d',
            in_channels = 256, 
            out_channels = 64, 
            hidden_channels = 128,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_2 = Block(
            layer_type = 'Conv2d',
            in_channels = 128, 
            out_channels = 32, 
            hidden_channels = 64,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_3 = Block(
            layer_type = 'Conv2d',
            in_channels = 64, 
            out_channels = 16, 
            hidden_channels = 32,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_4 = Block(
            layer_type = 'Conv2d',
            in_channels = 32, 
            out_channels = 16, 
            hidden_channels = 16,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.generate = Block(
            layer_type = 'Conv2d',
            in_channels = 16, 
            out_channels = 1,
            hidden_channels = 1,
            num_layers = 1,
            activation_type = 'Sigmoid',
            kernel_size = 3
        )
    def forward(self, x):
        
        in_1 = self.transform_input_1(x)
        out_1 = self.encode_1(in_1)
        skip_1 = out_1.detach().clone()
        out_1 = self.pool(out_1)
        
        out_2 = self.encode_2(out_1)
        skip_2 = out_2.detach().clone()
        out_2 = self.pool(out_2)
        
        out_3 = self.encode_3(out_2)
        skip_3 = out_3.detach().clone()
        out_3 = self.pool(out_3)
        
        out_4 = self.encode_4(out_3)
        skip_4 = out_4.detach().clone()
        out_4 = self.pool(out_4)
        
        out_5 = self.encode_5(out_4)
        out_5 = self.upscale(out_5)
        
        out_6 = self.decode_1(torch.cat((skip_4, out_5), 1))
        out_6 = self.upscale(out_6)
        
        out_7 = self.decode_2(torch.cat((skip_3, out_6), 1))
        out_7 = self.upscale(out_7)
        
        out_8 = self.decode_3(torch.cat((skip_2, out_7), 1))
        out_8 = self.upscale(out_8)
        
        out_9 = self.decode_4(torch.cat((skip_1, out_8), 1))
        
        out = self.generate(out_9)
        return self.transform_output(out)

class R2UNet(nn.Module):
    def __init__(self, input_size=(128, 128), output_size=(128, 128)):
        super().__init__()
        
        self.transform_input = nn.AdaptiveAvgPool2d(input_size)
        self.transform_output = nn.AdaptiveAvgPool2d(output_size)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.encode_1 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 1, 
            out_channels = 16,
            hidden_channels = 16,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_2 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 16, 
            out_channels = 32,
            hidden_channels = 32,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_3 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 32, 
            out_channels = 64, 
            hidden_channels = 64,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_4 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 64, 
            out_channels = 128, 
            hidden_channels = 128,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_5 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 128, 
            out_channels = 128, 
            hidden_channels = 256 ,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_1 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 256, 
            out_channels = 64, 
            hidden_channels = 128,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_2 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 128, 
            out_channels = 32, 
            hidden_channels = 64,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_3 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 64, 
            out_channels = 16, 
            hidden_channels = 32,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_4 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 32, 
            out_channels = 16, 
            hidden_channels = 16,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.generate = Block(
            layer_type = 'Conv2d',
            in_channels = 16, 
            out_channels = 1,
            hidden_channels = 1,
            num_layers = 1,
            activation_type = 'Sigmoid',
            kernel_size = 3
        )

    def forward(self, x):
        
        in_1 = self.transform_input(x)
        out_1 = self.encode_1(in_1)
        skip_1 = out_1.detach().clone()
        out_1 = self.pool(out_1)
        
        out_2 = self.encode_2(out_1)
        skip_2 = out_2.detach().clone()
        out_2 = self.pool(out_2)
        
        out_3 = self.encode_3(out_2)
        skip_3 = out_3.detach().clone()
        out_3 = self.pool(out_3)
        
        out_4 = self.encode_4(out_3)
        skip_4 = out_4.detach().clone()
        out_4 = self.pool(out_4)
        
        out_5 = self.encode_5(out_4)
        out_5 = self.upscale(out_5)
        
        out_6 = self.decode_1(torch.cat((skip_4, out_5), 1))
        out_6 = self.upscale(out_6)
        
        out_7 = self.decode_2(torch.cat((skip_3, out_6), 1))
        out_7 = self.upscale(out_7)
        
        out_8 = self.decode_3(torch.cat((skip_2, out_7), 1))
        out_8 = self.upscale(out_8)
        
        out_9 = self.decode_4(torch.cat((skip_1, out_8), 1))
        
        out = self.generate(out_9)

        return self.transform_output(out)

# U-Net with Recurrent Residual Blocks and spacial and channel attention (CBAM)
# Base U-Net Model: https://bmcbiomedeng.biomedcentral.com/articles/10.1186/s42490-021-00050-y
# Recurrent Residual Blocks: https://arxiv.org/abs/1802.06955
class R2UNet(nn.Module):
    def __init__(self, input_size=(128, 128), output_size=(128, 128)):
        super().__init__()
        
        self.transform_input = nn.AdaptiveAvgPool2d(input_size)
        self.transform_output = nn.AdaptiveAvgPool2d(output_size)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.encode_1 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 1, 
            out_channels = 16,
            hidden_channels = 16,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_2 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 16, 
            out_channels = 32,
            hidden_channels = 32,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_3 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 32, 
            out_channels = 64, 
            hidden_channels = 64,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_4 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 64, 
            out_channels = 128, 
            hidden_channels = 128,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.encode_5 = Block(
            layer_type = 'Conv2d',
            in_channels = 128, 
            out_channels = 128, 
            hidden_channels = 256 ,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_1 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 256, 
            out_channels = 64, 
            hidden_channels = 128,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_2 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 128, 
            out_channels = 32, 
            hidden_channels = 64,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_3 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 64, 
            out_channels = 16, 
            hidden_channels = 32,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.decode_4 = R2Block(
            layer_type = 'Conv2d',
            in_channels = 32, 
            out_channels = 16, 
            hidden_channels = 16,
            num_layers = 2,
            activation_type = 'ReLU'
        )
        
        self.generate = Block(
            layer_type = 'Conv2d',
            in_channels = 16, 
            out_channels = 1,
            hidden_channels = 1,
            num_layers = 1,
            activation_type = 'Sigmoid',
            kernel_size = 3
        )

    def forward(self, x):
        
        in_1 = self.transform_input(x)
        out_1 = self.encode_1(in_1)
        skip_1 = out_1.detach().clone()
        out_1 = self.pool(out_1)
        
        out_2 = self.encode_2(out_1)
        skip_2 = out_2.detach().clone()
        out_2 = self.pool(out_2)
        
        out_3 = self.encode_3(out_2)
        skip_3 = out_3.detach().clone()
        out_3 = self.pool(out_3)
        
        out_4 = self.encode_4(out_3)
        skip_4 = out_4.detach().clone()
        out_4 = self.pool(out_4)
        
        out_5 = self.encode_5(out_4)
        out_5 = self.upscale(out_5)
        
        out_6 = self.decode_1(torch.cat((skip_4, out_5), 1))
        out_6 = self.upscale(out_6)
        
        out_7 = self.decode_2(torch.cat((skip_3, out_6), 1))
        out_7 = self.upscale(out_7)
        
        out_8 = self.decode_3(torch.cat((skip_2, out_7), 1))
        out_8 = self.upscale(out_8)
        
        out_9 = self.decode_4(torch.cat((skip_1, out_8), 1))
        
        out = self.generate(out_9)

        return self.transform_output(out)