"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=12, padding=12)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(x))
        dilate3_out = nonlinearity(self.dilate3(x))
        dilate4_out = nonlinearity(self.dilate4(x))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out #+ dilate5_out
        return out


class LDblock(nn.Module):
    def __init__(self, channel):
        super(LDblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=12, padding=12)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class LDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(LDecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 2)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 2, n_filters, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_more_dilate, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


# class DinkNet34(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3):
#         super(DinkNet34, self).__init__()
#
#         filters = [64, 128, 256, 512]
#
#         resnet = models.resnet34(pretrained=True)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#         self.deep_conv = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1),
#                                          nn.BatchNorm2d(64),
#                                          nn.ReLU(),
#                                          nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
#         self.sample_conv = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1),
#                                           nn.BatchNorm2d(32),
#                                           nn.ReLU(),
#                                           nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))
#         self.dblock = Dblock(256)
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         # self.finalconv2 = nn.ConvTranspose2d(64, 32, 2, 2, 1)
#         # self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         # x = self.firstconv(x[:, 0:3, :, :])
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         # e4 = self.encoder4(e3)
#         deep = self.deep_conv(e3)
#         deep = F.upsample(deep, size=(512,512), mode='bilinear', align_corners=True)
#         # Center
#         # e4 = self.dblock(e4)
#         e3 = self.dblock(e3)
#         # Decoder
#         # d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(e3) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2+x)
#
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         # out = self.finalconv2(out)
#         # out = self.finalrelu2(out)
#         res = self.finalconv3(out)
#         lower = self.sample_conv(out)
#         return deep, res, lower


class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]

        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.sample_conv1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(),
                                          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.sample_conv2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(),
                                          nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.sample_conv3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

        self.deep_conv = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
        self.sample_conv = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))
        self.dblock = Dblock(256)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.ConvTranspose2d(64, 32, 2, 2, 1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(128, momentum=0.01), nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(64, momentum=0.01), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(32, momentum=0.01), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16, momentum=0.01), nn.ReLU())
        self.segment = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        l1 = self.firstconv(x)
        # x = self.firstconv(x[:, 0:3, :, :])
        l1 = self.firstbn(l1)
        l1 = self.firstrelu(l1)
        l1 = self.firstmaxpool(l1)
        e1 = self.encoder1(l1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        e1 = self.sample_conv1(e1)
        e2 = self.sample_conv2(e2)
        e3 = self.sample_conv3(e3)

        # e4 = self.encoder4(e3)
        deep = self.deep_conv(e3)
        deep = F.upsample(deep, size=x.size()[2:], mode='bilinear', align_corners=True)
        # Center
        # e4 = self.dblock(e4)
        db = self.dblock(e3)
        # Decoder
        # d4 = self.decoder4(e4) + e3
        dec = self.decoder(db)
        seg = self.segment(dec)

        d3 = self.decoder3(db) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2+l1)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        ## out = self.finalconv2(out)
        ## out = self.finalrelu2(out)
        # res = self.finalconv3(out)
        lower = self.sample_conv(out)

        return deep, seg, lower
        # return deep, lower, seg



class PositionAttentionModule(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x
        return out

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""
    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x
        return out

class SelfAttention(nn.Module):

    def __init__(self, inplanes):
        super(SelfAttention, self).__init__()
        self.inplanes = inplanes
        # theta transform
        self.theta = nn.Conv2d(self.inplanes, self.inplanes // 2, 1)
        # phi transform
        self.phi = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes // 2, 1),
                                 # nn.MaxPool2d(2,2)
                                 )
        # function g
        self.g_func = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes // 2, 1),
                                    # nn.MaxPool2d(2,2)
                                    )

        # w transform to match the channel
        self.w = nn.Conv2d(self.inplanes // 2, self.inplanes, 1)

    def forward(self, xs):
        theta = self.theta(xs)
        N, C, W, H = theta.size()
        theta = theta.view(N, C, H * W).transpose(2, 1)
        # print(theta.shape)
        phi = self.phi(xs)
        phi = phi.view(N, C, -1)
        # compute attention
        attention = theta.bmm(phi)
        assert attention.size() == (N, H * W, H * W)
        attention = nn.functional.softmax(attention, dim=-1)
        # g transform
        g = self.g_func(xs)
        g = g.view(N, C, -1)
        # final response
        response = g.bmm(attention.transpose(2, 1))
        response = response.view(N, C, W, H)
        # matching channel
        response = self.w(response)
        output = response + xs
        return output

class OurDinkNet50(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(OurDinkNet50, self).__init__()

        filters = [256, 512, 1024, 128]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.first_conv = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 256, kernel_size=1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(1024)
        self.pam3 = PositionAttentionModule(in_channels=1024)
        self.cam = ChannelAttentionModule()
        self.sample_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1))
        # self.merge_conv3 = nn.Conv2d(1024, 512, 1, padding=0)
        # self.merge_relu3 = nonlinearity
        # self.merge_conv2 = nn.Conv2d(512, 256, 1, padding=0)
        # self.merge_relu2 = nonlinearity
        # self.merge_conv1 = nn.Conv2d(256, 128, 1, padding=0)
        # self.merge_relu1 = nonlinearity

        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[3])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[3], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 1)

    def forward(self, input):
        # Encoder
        x = self.firstconv(input)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        # e4 = self.encoder4(e3)

        e3 = self.pam3(e3)
        e3 = self.dblock(e3)
        e4 = self.cam(e3)
        # Center
        # e4 = self.dblock(e3)

        # Decoder
        # d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(e4) + e2
        d2 = self.decoder2(d3) + e1
        x = self.first_conv(x)
        d1 = self.decoder1(d2+x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        res = self.finalconv3(out)
        refine = self.sample_conv(out)
        return res, refine
