# -*- coding: utf-8 -*-
# @Time    : 2023/3/4 9:14
# @Author  : Lin Junhong
# @FileName: test.py
# @Software: PyCharm
# @E_mails ：SPJLinn@163.com

import argparse
import os
import time

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as UT
from models.LDRM import LDRM as Network

# ======================================================================================================================
weights_dir = './checkpoints/Real.pth'
data_dir = './testset/Real/'
result_dir = './results/Real/'

if os.path.isfile(data_dir):
    Scenario = 'single'
elif os.path.isfile(data_dir + os.listdir(data_dir)[0]):
    Scenario = 'real'
else:
    Scenario = 'synth'

data_dir_str = data_dir.split('/')[2:-1]
data_dir_str = '/'.join(data_dir_str)
# resul_dir = result_dir + data_dir_str

parser = argparse.ArgumentParser(description='Low-light Imaging via Color-Monochrome Cameras')
parser.add_argument('--input_dir', default=data_dir, type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default=result_dir, type=str, help='Directory for results')
parser.add_argument('--weights_dir', default=weights_dir, type=str, help='Path to weights')
parser.add_argument('--scenario', default=Scenario, type=str, help='Different test Scenario')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# ======================================================================================================================
class TEST:
    def __init__(self, weights_dir):
        self.model = Network().to(args.device)
        UT.NET.load_checkpoint(self.model, weights_dir, strict=True)
        self.model.eval()
        # self.model = nn.DataParallel(self.model)
        # macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
        #                                          print_per_layer_stat=False, verbose=True)

        result_dir = os.path.join(args.result_dir)
        os.makedirs(result_dir, exist_ok=True)

        print('====> Scenario: ', Scenario)
        # print(f'====> Params: {params}  Macs: {macs}')
        print(f'====> Testing weights: {weights_dir}')
        print('====> Save Dir: ', result_dir)

    def single_img(self, data_dir, result_dir):
        img_name = data_dir.split('/')[-1].split('.')[0]
        color_img = Image.open(data_dir)
        mono_img = Image.open('./testset/Mono.jpg')

        color_img = TF.to_tensor(color_img)
        mono_img = TF.to_tensor(mono_img)

        H = color_img.shape[-2]
        W = color_img.shape[-1]
        if H % 32 != 0:
            new_H = H // 32 * 32
        else:
            new_H = H

        if W % 32 != 0:
            new_W = W // 32 * 32
        else:
            new_W = W

        with torch.no_grad():

            color_img = torch.unsqueeze(color_img, 0).to(args.device)
            mono_img = torch.unsqueeze(mono_img, 0).to(args.device)

            start_time = time.time()
            restored_img = self.model(color_img, mono_img)
            end_time = time.time()
            pred_time = end_time - start_time

            restored_img = TF.resize(restored_img[0], (new_H, new_W))
            restored_img = torch.clamp(restored_img, 0, 1).permute(1, 2, 0).cpu().detach().numpy()
            UT.save_img((os.path.join(result_dir, img_name + '_Restored.png')), img_as_ubyte(restored_img))

        print(f"One Img Time: {pred_time}")

    def norefer(self, data_dir, result_dir):
        test_dataset = UT.DataLoaderTest(data_dir, ColorSet='Color')
        test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                                 shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
        print('Img Number: ', len(test_loader))

        with torch.no_grad():
            time_count = 0

            pbar = tqdm(enumerate(test_loader))
            for ii, data in pbar:
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                color = data[0].to(args.device)
                mono = data[1].to(args.device)

                start_time = time.time()
                restored = self.model(color, mono)
                end_time = time.time()
                pred_time = end_time - start_time

                filenames = data[2]
                H, W = int(data[3][0]), int(data[4][0])

                restored = TF.resize(restored, (H, W), interpolation=PIL.Image.BICUBIC)
                restored = torch.clamp(restored, 0, 1).permute(0, 2, 3, 1).cpu().detach().numpy()

                time_count += pred_time

                for batch in range(len(restored)):
                    UT.save_img((os.path.join(result_dir, f'{filenames[batch]}_Restored.png')), img_as_ubyte(restored[batch]))
                info = f'One Img Time:{pred_time:.4f} Total Processing Time:{time_count:.4f} '
                pbar.set_description(info)

        img_per_sec = time_count / (len(test_loader))
        print(f"Avg One Img Time: {img_per_sec}")

    def refer(self, data_dir, result_dir):
        # img_options={'patch_size': 256}
        val_dataset = UT.DataLoaderVal(data_dir, patch_size=None,
                                       Scale=1, ColorSet='Color/')
        val_loader = DataLoader(dataset=val_dataset, batch_size=1,
                                shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
        print('Img Number: ', len(val_loader) * 1)

        with torch.no_grad():
            psnr_val_rgb = []
            ssim_val_rgb = []
            time_count = 0
            PSNRS = 0
            SSIMS = 0
            pbar = tqdm(enumerate(val_loader))
            for ii, data in pbar:
                color = data[0].to(args.device)
                mono = data[1].to(args.device)
                gt = data[2].to(args.device)
                filenames = data[3]
                H, W = int(data[4][0]), int(data[5][0])

                start_time = time.time()
                restored = self.model(color, mono)
                end_time = time.time()
                pred_time = end_time - start_time
                restored = TF.resize(restored, (H, W), interpolation=PIL.Image.BICUBIC)
                restored = torch.clamp(restored, 0, 1).cpu().detach().numpy().squeeze().transpose(1, 2, 0)
                gt = torch.clamp(gt, 0, 1).cpu().detach().numpy().squeeze().transpose(1, 2, 0)
                time_count = time_count + pred_time
                # for batch in range(len(clean)):
                psnr = psnr_loss(restored, gt)
                # ssim = ssim_loss(restored, gt, win_size=3, multichannel=True)
                PSNRS += psnr
                # SSIMS += ssim
                psnr_val_rgb.append(psnr)
                # ssim_val_rgb.append(ssim)
                # print(restored.shape)
                UT.save_img((os.path.join(result_dir, filenames[0] + '_Restored.png')), img_as_ubyte(restored))

                info = f'One Batch Time:{pred_time:.4f} Image:{H, W} PSNR:{psnr:.4f}' \
                       f'Total Processing Time:{time_count:.4f} '
                pbar.set_description(info)
            # print(f'Avg PSNR: {PSNRS/len(val_loader)} SSIM: {SSIMS/len(val_loader)}')
        img_per_sec = time_count / (len(val_loader) * 1)
        psnr_val_rgb = sum(psnr_val_rgb) / (len(val_loader) * 1)
        # ssim_val_rgb = sum(ssim_val_rgb) / (len(val_loader) * 1)
        print(f"Avg One Img Time: {img_per_sec} \nPSNR: {psnr_val_rgb}  SSIM: {ssim_val_rgb}")


# ======================================================================================================================
if __name__ == '__main__':
    T = TEST(weights_dir=weights_dir)
    T.refer(data_dir, result_dir)
