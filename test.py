import os
import sys
import numpy as np
import torch
import time
from thop import profile
from thop import clever_format
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Testmodel

from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='', help='location of the data corpus')
parser.add_argument('--model', type=str, default='', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def rgb2yCbC(input_im, path):
    im_flat = input_im.contiguous().view(-1, 3).float()
    mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
    bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
    temp = im_flat.mm(mat) + bias
    out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
    return out


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Testmodel(args.model)
    model = model.cuda()

    model.eval()

    # Computer FLOPS AND PARAMERTERS
    # inputs = torch.randn(1, 3, 540, 960)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = inputs.to(device)
    # flops, params = profile(model, inputs=(inputs,))
    # flops, params = clever_format([flops, params], '%.3f')
    # print('FLOPs: %s' % (flops))
    # print('params:%s' % (params))

    runTime=[]
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input).cuda()
            # image_name = image_name[0].split('/')[-1].split('.')[0]
            newname = image_name[0].split('/')[-1].strip('.jpg\n')
            # illu_list, ref_list, input_list, atten = model(input)


            #COMPUTER RUNNING TIME
            # start = time.clock()
            illu_list, ref_list = model(input)
            # end = time.clock()
            # runTime.append(end - start)
            # print("nfer time:", runTime)
            # u_name = '%s.png' % (image_name)
            u_name = '%s.png' % (newname)
            print('processing {}'.format(u_name))
            u_path = save_path + u_name
            # out_ill = rgb2yCbC(illu_list, u_path)
            save_images(illu_list, u_path)
            # save_images(ref_list, u_path)
        # print('infer time:', np.mean(runTime[1:]))



if __name__ == '__main__':
    main()
