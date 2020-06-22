import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import torchvision.transforms as transforms
from PIL import Image


class Generator_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.tahn = nn.Tanh()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            #nn.Conv2d(3, 32, (7, 7), stride=1, padding = (3,3), padding_mode = 'reflect'),
            nn.Conv2d(3, 32, (7, 7), stride=1, padding = (3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), stride=2, padding = (1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=2, padding = (1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        # 64, 64, 128

        # transform
        self.transform1 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.BatchNorm2d(128)
        )

        self.transform2 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.BatchNorm2d(128)
        )

        self.transform3 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.BatchNorm2d(128)
        )

        self.transform4 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.BatchNorm2d(128)
        )

        self.transform5 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.BatchNorm2d(128)
        )

        self.transform6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.BatchNorm2d(128)
        )

        self.transform7 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.BatchNorm2d(128)
        )

        self.transform8 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.BatchNorm2d(128)
        )

        self.transform9 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=(1,1), padding_mode = 'reflect'),
            nn.BatchNorm2d(128)
        )
        # 64, 64, 64

        # decoder (upsampling)
        self.de_conv0 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = (1,1), output_padding = (1,1)),# 16 -> 32
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.de_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride = 2, padding = (1,1), output_padding = (1,1)), # 32 -> 64
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.de_conv2 = nn.Sequential(
            nn.Conv2d(32, 3, (7, 7), padding = (3,3)),
            nn.BatchNorm2d(3)
        )


    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(e0)
        e2 = self.enc_conv2(e1)

        # transform
        b0_0 = self.transform1(e2)
        #b0_1 = self.relu(torch.cat([e2, b0_0], 1))
        b0_1 = self.relu(e2 + b0_0)

        b1_0 = self.transform2(b0_1)
        #b1_1 = self.relu(torch.cat([b0_1, b1_0], 1))
        b1_1 = self.relu(b0_1 + b1_0)

        b2_0 = self.transform3(b1_1)
        #b2_1 = self.relu(torch.cat([b1_1, b2_0], 1))
        b2_1 = self.relu(b1_1 + b2_0)

        b3_0 = self.transform4(b2_1)
        #b3_1 = self.relu(torch.cat([b2_1, b3_0], 1))
        b3_1 = self.relu(b2_1 + b3_0)

        b4_0 = self.transform5(b3_1)
        #b4_1 = self.relu(torch.cat([b3_1, b4_0], 1))
        b4_1 = self.relu(b3_1 + b4_0)

        b5_0 = self.transform6(b4_1)
        #b5_1 = self.relu(torch.cat([b4_1, b5_0], 1))
        b5_1 = self.relu(b4_1 + b5_0)

        b6_0 = self.transform7(b5_1)
        #b3_1 = self.relu(torch.cat([b2_1, b3_0], 1))
        b6_1 = self.relu(b5_1 + b6_0)

        b7_0 = self.transform8(b6_1)
        #b4_1 = self.relu(torch.cat([b3_1, b4_0], 1))
        b7_1 = self.relu(b6_1 + b7_0)

        b8_0 = self.transform9(b7_1)
        #b5_1 = self.relu(torch.cat([b4_1, b5_0], 1))
        b8_1 = self.relu(b7_1 + b8_0)

        # decoder
        d0 = self.de_conv0(b8_1)
        d1 = self.de_conv1(d0)
        d2 = self.de_conv2(d1)

        d3 = self.tahn(d2)

        return d3

class Discriminator_net(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.tahn = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, (4, 4), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (4, 4)),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 1, (4, 4))
        )

    def RandomCrop(x, size, side_size):
        stop = x.shape[2] - side_size
        start = random.randint(side_size, stop - size)

        x = x[:, :, start:start + size, start:start + size]
        return x

    def forward(self, x):
        # encoder
        x = RandomCrop(x, 70, 55)
        e0 = self.conv0(x)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)

        # d3 = self.tahn(e4)

        return e4

class Transform_model():

    def __init__(self, path_name, state_dir):
        # все изображения будут масштабированы к размеру 1024x**** px
        # иначе расчеты идут ну очень долго
        self.RESCALE_SIZE = 360
        self.save_path = path_name
        self.gen_A = Generator_net()
        self.gen_B = Generator_net()

        self.dec_A = Discriminator_net()
        self.dec_B = Discriminator_net()

        self.gen_A.load_state_dict(torch.load(self.save_path + state_dir + 'BCE-gen_A.pt', map_location='cpu'))
        self.gen_B.load_state_dict(torch.load(self.save_path + state_dir + 'BCE-gen_B.pt', map_location='cpu'))
        self.dec_A.load_state_dict(torch.load(self.save_path + state_dir + 'BCE-dec_A.pt', map_location='cpu'))
        self.dec_B.load_state_dict(torch.load(self.save_path + state_dir + 'BCE-dec_B.pt', map_location='cpu'))

        self.gen_A.eval()
        self.gen_B.eval()
        self.dec_A.eval()
        self.dec_B.eval()

    def img_to_tensor(self, file_name):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = Image.open(self.save_path + file_name)
        image.load()

        #image = image.resize((self.RESCALE_SIZE, self.RESCALE_SIZE))
        curr_size = image.size
        if curr_size[0] > self.RESCALE_SIZE:
            size0 = self.RESCALE_SIZE
            size1 = int(curr_size[1] / curr_size[0] * self.RESCALE_SIZE)
            image = image.resize((size0, size1))
            
        image = np.array(image)

        image = np.array(image / 127.5 - 1, dtype='float32')
        A = transform(image)

        return A

    def tensor_to_img(self, res_tensor, file_name):
        res_img = res_tensor.detach().numpy().transpose((1, 2, 0))
        res_img = ((res_img + 1) * 127.5).astype(np.uint8)
        result = Image.fromarray(res_img)
        result.save(self.save_path + 'conv_' + file_name)
        #return result
        return self.save_path + 'conv_' + file_name

    def Transform_to_B(self, file_name):
        input_img = self.img_to_tensor(file_name)
        input_img = input_img.unsqueeze(0)
        val_B_gen = self.gen_B(input_img)
        return self.tensor_to_img(val_B_gen[0],file_name)

    def Transform_to_A(self, file_name):
        input_img = self.img_to_tensor(file_name)
        input_img = input_img.unsqueeze(0)
        val_A_gen = self.gen_A(input_img)
        return self.tensor_to_img(val_A_gen[0],file_name)
