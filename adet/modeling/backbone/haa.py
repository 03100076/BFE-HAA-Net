import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAtt(nn.Module):
    def __init__(self, channels):
        super(ChannelAtt, self).__init__()
        self.channels = channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1)
        )        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight1 = self.conv1(self.avgpool(x))
        weight2 = self.conv2(self.maxpool(x))
        weight = self.sigmoid(weight1 + weight2)
        
        output = x * weight
        return output

# 1 page
class ChannelAttAvg(nn.Module):
    def __init__(self, channels):
        super(ChannelAttAvg, self).__init__()
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1)
        )      
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight = self.conv(self.avgpool(x))
        weight = self.sigmoid(weight)
        
        output = x * weight
        return output
 
# 2 page    
class ChannelAttMax(nn.Module):
    def __init__(self, channels):
        super(ChannelAttMax, self).__init__()
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1)
        )      
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight = self.conv(self.maxpool(x))
        weight = self.sigmoid(weight)
        
        output = x * weight
        return output

# 3 page    
class ChannelAttCat(nn.Module):
    def __init__(self, channels):
        super(ChannelAttCat, self).__init__()
        self.channels = channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1)
        )
        self.conv3 = nn.Conv2d(in_channels=2 * channels, out_channels=channels, kernel_size=1)         
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight1 = self.conv1(self.avgpool(x))
        weight2 = self.conv2(self.maxpool(x))
        weight = self.sigmoid(self.conv3(torch.cat([weight1, weight2], dim=1)))
        
        output = x * weight
        return output

# 4 page    
class ChannelAttCatFirst(nn.Module):
    def __init__(self, channels):
        super(ChannelAttCatFirst, self).__init__()
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * channels, out_channels=channels//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight = self.conv(torch.cat([self.avgpool(x), self.maxpool(x)], dim=1))
        weight = self.sigmoid(weight)
        
        output = x * weight
        return output


class SpaceAtt(nn.Module):
    def __init__(self):
        super(SpaceAtt, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight1 = torch.mean(x, dim=1, keepdim=True)
        weight2 = torch.max(x, dim=1, keepdim=True)[0]
        weight3 = self.conv1(torch.cat([weight1, weight2], dim=1))
        weight = self.conv2(weight1 + weight2 + weight3)
        
        output = x * weight
        return output

# 5 page    
class SpaceAttAvg(nn.Module):
    def __init__(self):
        super(SpaceAttAvg, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight = torch.mean(x, dim=1, keepdim=True)
        weight = self.conv(weight)
        
        output = x * weight
        return output

# 6 page    
class SpaceAttMax(nn.Module):
    def __init__(self):
        super(SpaceAttMax, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight = torch.max(x, dim=1, keepdim=True)[0]
        weight = self.conv(weight)
        
        output = x * weight
        return output

# 7 page    
class SpaceAttSum(nn.Module):
    def __init__(self):
        super(SpaceAttSum, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight1 = torch.mean(x, dim=1, keepdim=True)
        weight2 = torch.max(x, dim=1, keepdim=True)[0]
        weight = self.conv(weight1 + weight2)
        
        output = x * weight
        return output

# 8 page    
class SpaceAttCat(nn.Module):
    def __init__(self):
        super(SpaceAttCat, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight1 = torch.mean(x, dim=1, keepdim=True)
        weight2 = torch.max(x, dim=1, keepdim=True)[0]
        weight = self.conv(torch.cat([weight1, weight2], dim=1))
        
        output = x * weight
        return output
# 9 page    
class SpaceAttDown(nn.Module):
    def __init__(self, channels):
        super(SpaceAttDown, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, groups=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1, stride=1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # x.shape: (N, C, H, W)
        _, _, target_h, target_w = x.size()
        feature = self.conv1(x)
        
        feature = F.interpolate(
            feature,
            size=(target_h, target_w),
            mode="nearest")
        
        weight = self.conv2(feature)
        
        output = x * weight
        return output
    
# 10 page    
class SpaceAttDownPoolCat(nn.Module):
    def __init__(self, channels):
        super(SpaceAttDownPoolCat, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, groups=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1, stride=1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # x.shape: (N, C, H, W)
        _, _, target_h, target_w = x.size()
        feature = self.conv1(x)
        
        feature = F.interpolate(
            feature,
            size=(target_h, target_w),
            mode="nearest")
        weight1 = torch.mean(feature, dim=1, keepdim=True)
        weight2 = torch.max(feature, dim=1, keepdim=True)[0]

        weight = self.conv2(torch.cat([weight1, weight2], dim=1))
        
        output = x * weight
        return output
    
# 11 page    
class SpaceAttDownPoolSum(nn.Module):
    def __init__(self, channels):
        super(SpaceAttDownPoolSum, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, groups=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1, stride=1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # x.shape: (N, C, H, W)
        _, _, target_h, target_w = x.size()
        feature = self.conv1(x)
        
        feature = F.interpolate(
            feature,
            size=(target_h, target_w),
            mode="nearest")
        weight1 = torch.mean(feature, dim=1, keepdim=True)
        weight2 = torch.max(feature, dim=1, keepdim=True)[0]

        weight = self.conv2(weight1 + weight2)
        
        output = x * weight
        return output
    
class ChannelAttSum(nn.Module):
    def __init__(self, channels):
        super(ChannelAttSum, self).__init__()
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (N, C, H, W)
        weight1 = self.conv(self.avgpool(x))
        weight2 = self.conv(self.maxpool(x))
        weight = self.sigmoid(weight1 + weight2)
        
        output = x * weight
        return output

# 12 page
class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.branch1 = SpaceAttCat()
        self.branch2 = ChannelAttSum(channels)
    
    def forward(self, x):
        # x.shape: (N, C, H, W)
        output = self.branch2(x)
        output = self.branch1(output)
        return output
    
# 13 page
class F1(nn.Module):
    def __init__(self, channels):
        super(F1, self).__init__()
        self.branch1 = SpaceAtt()
        self.branch2 = ChannelAtt(channels)
    
    def forward(self, x):
        # x.shape: (N, C, H, W)
        output = self.branch2(x)
        output = self.branch1(output)
        return output

class HAANet(nn.Module):
    def __init__(self, channels):
        super(HAANet, self).__init__()
        self.branch1 = SpaceAtt()
        self.branch2 = ChannelAtt(channels)
    
    def forward(self, x):
        # x.shape: (N, C, H, W)
        output1 = self.branch1(x)
        output2 = self.branch2(x)
        output = output1 + output2
        return output

if __name__ == '__main__':
    x = torch.randn(2, 256, 100, 80)
    ca = ChannelAtt(256)
    sa = SpaceAtt()
    print(ca(x).shape)
    print(sa(x).shape)
    

# Example usage:
# model = SimpleCNN(num_classes=10)
# print(model)
