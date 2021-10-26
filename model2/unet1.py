import torchvision
from unet2 import *
from torchsummary import summary

class Unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        # for i in range(len(self.base_layers)):
        #     print(self.base_layers[i])
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
            self.base_layers[4]
        )
        self.att1 = CA_Block(64, 256, 256)

        self.layer2 = self.base_layers[5]
        self.att2 = CA_Block(128, 128, 128)

        self.layer3 = self.base_layers[6]
        self.att3 = CA_Block(256, 64, 64)

        self.layer4 = self.base_layers[7]
        self.att4 = CA_Block(512, 32, 32)

        self.layer5 = ASPP(512, 512)

        self.decode4 = Decoder(512, 512 + 512, 512)
        self.decode3 = Decoder(512, 256 + 256, 256)
        self.decode2 = Decoder(256, 128 + 128, 128)
        self.decode1 = Decoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e1 = self.layer1(input)  # 64,256,256
        x1_a = self.att1(e1)

        e2 = self.layer2(e1)  # 128,128,128
        x2_a = self.att2(e2)

        e3 = self.layer3(e2)  # 256,64,64
        x3_a = self.att3(e3)

        e4 = self.layer4(e3)  # 512,32,32
        x4_a = self.att4(e4)

        f = self.layer5(e4)  # 512,16,16

        d4 = self.decode4(f, x4_a)  # 256,16,16
        d3 = self.decode3(d4, x3_a)  # 256,32,32
        d2 = self.decode2(d3, x2_a)  # 128,64,64
        d1 = self.decode1(d2, x1_a)  # 64,128,128
        d0 = self.decode0(d1)  # 64,256,256
        out = self.conv_last(d0)  # 1,256,256
        return out

if __name__ == '__main__':
    # net = UNet(n_channels=3, n_classes=1)
    # print(net)
    unet = Unet(1)
    # print(unet)
    # summary(unet, input_size=(1, 512, 512), device='cpu')
    x = torch.randn(1, 1, 512, 512)
    out=unet(x)
    print('输出：')
    print(out.shape)