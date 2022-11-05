import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util import conv, predict_flow, deconv, crop_like

__all__ = [
    'flownets', 'flownets_bn'
]

class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetS, self).__init__()


        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4 = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5 = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.deconv1 = deconv(194, 32)
        self.deconv0 = deconv(98, 16)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(98)
        self.predict_flow0 = predict_flow(24)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x, flipInput=False):
        
        frame0 = x[0]  # c, h, w
        frame1 = x[1]  # c, h, w

        if flipInput:
            combined = torch.cat([frame1, frame0], 0).unsqueeze(dim=0) # b, 6, 224, 224
        else:
            combined = torch.cat([frame0, frame1], 0).unsqueeze(dim=0) # b, 6, 224, 224

        out_conv1 = self.conv1(combined) # 1, 64, 112, 112
        out_conv2 = self.conv2(out_conv1) # 1, 128, 56, 56
        out_conv3 = self.conv3_1(self.conv3(out_conv2))  # 1, 256, 28, 28
        out_conv4 = self.conv4_1(self.conv4(out_conv3)) # 1, 512, 14, 14
        out_conv5 = self.conv5_1(self.conv5(out_conv4)) # 1, 512, 7, 7
        out_conv6 = self.conv6_1(self.conv6(out_conv5)) # 1, 1024, 4, 4

        flow6 = self.predict_flow6(out_conv6) # 1, 2, 4, 4
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5) # 1, 2, 7, 7
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5) # 1, 512, 7, 7

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1) # 1, 1026, 7, 7
        flow5 = self.predict_flow5(concat5) # 1, 2, 7, 7
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4) # 1, 2, 14, 14
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4) # 1, 256, 14, 14

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1) # 1, 770, 14, 14
        flow4 = self.predict_flow4(concat4) # 1, 2, 14, 14
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3) # 1, 2, 28, 28
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3) # 1, 128, 28, 28

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1) # 1, 386, 28, 28
        flow3 = self.predict_flow3(concat3) # 1, 2, 28, 28
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2) # 1, 2, 56, 56
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2) # 1, 64, 56, 56

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1) # 1, 194, 56, 56
        flow2 = self.predict_flow2(concat2) # 1, 2, 56, 56
        flow2_up = crop_like(self.upsampled_flow2_to_1(flow2), out_conv1) # 1, 2, 112, 112
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1) # 1, 32, 112, 112

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1) # 1, 98, 112, 112
        flow1 = self.predict_flow1(concat1) # 1, 2, 112, 112
        flow1_up = crop_like(self.upsampled_flow1_to_0(flow1), combined) # 1, 2, 224, 224
        out_deconv0 = crop_like(self.deconv0(concat1), combined) # 1, 16, 224, 224

        concat0 = torch.cat((combined, out_deconv0, flow1_up), 1) # 1, 24, 224, 224
        flow0 = self.predict_flow0(concat0) # 1, 2, 224, 224

        predict_flow0 = flow0 * 20

        if self.training:
            # return flow0, flow1, flow2, flow3, flow4, flow5, flow6
            return predict_flow0
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownets(data=None, batchNorm=False):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=batchNorm)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def flownets_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
