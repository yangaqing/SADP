import math
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import utils_image as util
import os
import cv2
from torchvision import transforms

import utils
import torchvision.models as models
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from model import *
from multi_read_data import MemoryFriendlyLoader
# from finetune import pic_name
sys.path.append('/home/yaq/Downloads/Methods/SADP-main/scripts')
from val import get_segmentation_model


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    def showimg(self, input):
        if len(input.shape) == 4:
            RGB = (input * 255).byte()
            RGB = RGB.clone().detach().cpu().numpy()
            RGB1 = RGB[0].transpose(1, 2, 0)
            img1 = Image.fromarray(RGB1)
            plt.figure(1)
            plt.imshow(img1)
            plt.show()
        else:
            RGB = (input * 255).byte()
            RGB = RGB.clone().detach().cpu().numpy()
            # RGB1 = RGB[0].transpose(1, 2, 0)
            img1 = Image.fromarray(RGB)
            plt.figure(1)
            plt.imshow(img1)
            plt.show()

    def plot(self, y, name):
        plt.figure(num=name)
        plt.bar([i for i in range(len(y))], y, width=1)



    def save_images(self, tensor, path):
        image_numpy = tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
        im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
        im.save(path, 'png')


    def semanticloss(self, input, illu):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        BatchNorm2d = nn.BatchNorm2d
        fcnmodel = get_segmentation_model(model='fcn8s', dataset='pascal_voc', backbone = 'vgg16',
                                              aux=False, jpu = False, norm_layer=BatchNorm2d).to(device)

        weight = torch.load('/home/yaq/.torch/models/fcn8s_vgg16_pascal_voc_best_model232_gradient.pth')
        fcnmodel.load_state_dict(weight)
        fcnmodel.eval()
        illuimg = input_transform((illu.squeeze(0)*255).byte().clone().detach().cpu().numpy().transpose(1, 2, 0))
        inputimg = input_transform((input.squeeze(0)*255).byte().clone().detach().cpu().numpy().transpose(1, 2, 0))
        output_ill = fcnmodel(illuimg.unsqueeze(0).to(device))
        output_input = fcnmodel(inputimg.unsqueeze(0).to(device))
        predict = torch.argmax(output_ill[0].long(), 1) + 1
        target = torch.argmax(output_input[0].long(), 1) + 1
        pixel_labeled = torch.sum(predict > 0).item()
        pixel_wrong = torch.sum((predict != target) * (target > 0)).item()
        assert pixel_wrong <= pixel_labeled, "Correct area should be smaller than Labeled"
        sematic_loss = pixel_wrong / pixel_labeled
        return sematic_loss

    def extract_inter_feature(self, input, layer_name):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # normalize = transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        # transform = transforms.Compose([normalize])
        # img = transform(input)
        model = models.vgg16(pretrained=True).cuda()
        # input = torch.unsqueeze(input, 0).to(device)
        features_in_hook = []
        features_out_hook = []


        def hook(module, fea_in, fea_out):
            features_in_hook.append(fea_in)
            features_out_hook.append(fea_out)
            return None

        for (name, module) in model.named_modules():
            # print(name)
            if name in layer_name:
                module.register_forward_hook(hook=hook)
        model(input)
        return features_out_hook


    def vggloss(self, input, ref):
        # layer_name = ['features.3', 'features.8', 'features.15', 'features.22', 'features.29']
        layer_name = ['features.29']
        input_fea = self.extract_inter_feature(input, layer_name)
        ref_fea = self.extract_inter_feature(ref, layer_name)
        loss = nn.L1Loss()
        fea_loss = 0
        for i in range(len(input_fea)):
            fea_loss += loss(input_fea[i], ref_fea[i])
        return fea_loss

    def gradient_1orderloss(self, ref):
        gloss = 0
        x = []
        y = []
        RGB = (ref * 255).byte()
        RGB = RGB.clone().detach().cpu().numpy()
        if len(ref.shape) == 4:
            for i in range(ref.shape[0]):
                RGB1 = RGB[i].transpose(1, 2, 0)
                img = Image.fromarray(RGB1).convert('L')
                img = np.array(img)
                dx, dy = np.gradient(img/255, edge_order=1)
                x.append(dx)
                y.append(dy)
            if ref.shape[0] == 1:
                dx = torch.tensor(abs(x[0]))
                dy = torch.tensor(abs(y[0]))
            else:
                dx = torch.cat([torch.tensor(abs(x[0])).unsqueeze(0), torch.tensor(abs(x[1])).unsqueeze(0)],0)
                dy = torch.cat([torch.tensor(abs(y[0])).unsqueeze(0), torch.tensor(abs(y[1])).unsqueeze(0)],0)
            gloss = torch.mean((dx+dy))
        return gloss






    def forward(self, input, illu, ref, input0):
        Fidelity_Loss = self.l2_loss(illu, input)
        Smooth_Loss = self.smooth_loss(input, illu)
        semantic_loss = self.semanticloss(input, illu)
        fea_loss = self.vggloss(input0, ref)
        gradloss = self.gradient_1orderloss(ref)
        return 1.5*Fidelity_Loss + Smooth_Loss + fea_loss + gradloss

     

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    # output: output      input:input
    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1,
                                        keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1,
                                        keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1,
                                        keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1,
                                        keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1,
                                        keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1,
                                        keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1,
                                        keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1,
                                        keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1,
                                        keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1,
                                        keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1,
                                        keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1,
                                        keepdim=True)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term
