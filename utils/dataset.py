import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'tif'])


class DataLoaderTrain(Dataset):
    def __init__(self, data_dir, patch_size=None, aug=False, Scale=1, Data='Syn', ColorSet='Color/X4'):
        super().__init__()
        assert os.path.exists(data_dir)
        self.Data = Data
        self.Scale = Scale

        color_files = sorted(os.listdir(os.path.join(data_dir, ColorSet)))
        mono_files = sorted(os.listdir(os.path.join(data_dir, 'Mono')))
        gt_files = sorted(os.listdir(os.path.join(data_dir, 'GT')))

        self.color_filenames = [os.path.join(data_dir, ColorSet, x) for x in color_files if is_image_file(x)]
        self.mono_filenames = [os.path.join(data_dir, 'Mono', x) for x in mono_files if is_image_file(x)]
        self.gt_filenames = [os.path.join(data_dir, 'GT', x) for x in gt_files if is_image_file(x)]

        self.patch_size = patch_size
        self.size = len(self.gt_filenames)  # get the size of Gt
        self.aug = aug

        if self.Data == 'Syn':
            cn_files = sorted(os.listdir(os.path.join(data_dir, f'Noise/{ColorSet}')))
            mn_files = sorted(os.listdir(os.path.join(data_dir, 'Noise/Mono')))
            self.cn_filenames = [os.path.join(data_dir, f'Noise/{ColorSet}', x) for x in cn_files if is_image_file(x)]
            self.mn_filenames = [os.path.join(data_dir, 'Noise/Mono', x) for x in mn_files if is_image_file(x)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index_ = index % self.size
        color_path = self.color_filenames[index_]
        mono_path = self.mono_filenames[index_]
        gt_path = self.gt_filenames[index_]

        color_path1 = open(color_path, 'rb')
        mono_path1 = open(mono_path, 'rb')
        gt_path1 = open(gt_path, 'rb')
        color_img = Image.open(color_path1)
        mono_img = Image.open(mono_path1)
        gt_img = Image.open(gt_path1)
        color_img = TF.to_tensor(color_img)
        mono_img = TF.to_tensor(mono_img)
        gt_img = TF.to_tensor(gt_img)

        if self.patch_size is not None:
            p_H, p_W = self.patch_size
            H = gt_img.shape[-2]
            W = gt_img.shape[-1]
            padw = p_W-W if W < p_W else 0
            padh = p_H-H if H < p_H else 0

            # Reflect Pad in case image is smaller than patch_size
            if padw != 0 or padh != 0:
                color_img = TF.pad(color_img, (0, 0, padw//self.Scale, padh//self.Scale), padding_mode='reflect')
                mono_img = TF.pad(mono_img, (0, 0, padw, padh), padding_mode='reflect')
                gt_img = TF.pad(gt_img, (0, 0, padw, padh), padding_mode='reflect')

            hh, ww = gt_img.shape[1], gt_img.shape[2]

            rr = random.randint(0, hh-p_H)
            cc = random.randint(0, ww-p_W)

            # Crop patch
            color_img = color_img[:, rr//self.Scale:(rr+p_H)//self.Scale, cc//self.Scale:(cc+p_W)//self.Scale]
            mono_img = mono_img[:, rr:rr+p_H, cc:cc+p_W]
            gt_img = gt_img[:, rr:rr+p_H, cc:cc+p_W]

        # Data Augmentations
        if self.aug:
            aug = random.randint(0, 8)
            if aug == 1:
                color_img = color_img.flip(1)
                mono_img = mono_img.flip(1)
                gt_img = gt_img.flip(1)
            elif aug == 2:
                color_img = color_img.flip(2)
                mono_img = mono_img.flip(2)
                gt_img = gt_img.flip(2)
            elif aug == 3:
                color_img = torch.rot90(color_img,dims=(1,2))
                mono_img = torch.rot90(mono_img,dims=(1,2))
                gt_img = torch.rot90(gt_img,dims=(1,2))
            elif aug == 4:
                color_img = torch.rot90(color_img,dims=(1,2), k=2)
                mono_img = torch.rot90(mono_img,dims=(1,2), k=2)
                gt_img = torch.rot90(gt_img,dims=(1,2), k=2)
            elif aug == 5:
                color_img = torch.rot90(color_img,dims=(1,2), k=3)
                mono_img = torch.rot90(mono_img,dims=(1,2), k=3)
                gt_img = torch.rot90(gt_img,dims=(1,2), k=3)
            elif aug == 6:
                color_img = torch.rot90(color_img.flip(1),dims=(1,2))
                mono_img = torch.rot90(mono_img.flip(1),dims=(1,2))
                gt_img = torch.rot90(gt_img.flip(1),dims=(1,2))
            elif aug == 7:
                color_img = torch.rot90(color_img.flip(2),dims=(1,2))
                mono_img = torch.rot90(mono_img.flip(2),dims=(1,2))
                gt_img = torch.rot90(gt_img.flip(2),dims=(1,2))

        filename = os.path.splitext(os.path.split(gt_path)[-1])[0]
        color_path1.close()
        mono_path1.close()
        gt_path1.close()

        if self.Data == 'Syn':
            cn_path = self.cn_filenames[index_]
            mn_path = self.mn_filenames[index_]

            cn_path1 = open(cn_path, 'rb')
            mn_path1 = open(mn_path, 'rb')
            cn_img = Image.open(cn_path1)
            mn_img = Image.open(mn_path1)
            cn_img = TF.to_tensor(cn_img)
            mn_img = TF.to_tensor(mn_img)

            if self.patch_size is not None:
                p_H, p_W = self.patch_size
                H = gt_img.shape[-2]
                W = gt_img.shape[-1]
                padw = p_W - W if W < p_W else 0
                padh = p_H - H if H < p_H else 0

                # Reflect Pad in case image is smaller than patch_size
                if padw != 0 or padh != 0:
                    cn_img = TF.pad(cn_img, (0, 0, padw//self.Scale, padh//self.Scale), padding_mode='reflect')
                    mn_img = TF.pad(mn_img, (0, 0, padw, padh), padding_mode='reflect')

                hh, ww = cn_img.shape[1], cn_img.shape[2]

                rr = random.randint(0, hh - p_H)
                cc = random.randint(0, ww - p_W)

                # Crop patch
                cn_img = cn_img[:, rr // self.Scale:(rr + p_H) // self.Scale,
                            cc // self.Scale:(cc + p_W) // self.Scale]
                mn_img = mn_img[:, rr:rr + p_H, cc:cc + p_W]

            # filename = os.path.splitext(os.path.split(cn_path1)[-1])[0]
            cn_path1.close()
            mn_path1.close()
            return color_img, mono_img, gt_img, cn_img, mn_img, filename
        else:
            return color_img, mono_img, gt_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, data_dir, patch_size=None, Scale=1, ColorSet='Color/X4'):
        super().__init__()
        assert os.path.exists(data_dir)
        self.Scale = Scale
        color_files = sorted(os.listdir(os.path.join(data_dir, ColorSet)))
        mono_files = sorted(os.listdir(os.path.join(data_dir, 'Mono')))
        gt_files = sorted(os.listdir(os.path.join(data_dir, 'GT')))

        self.color_filenames = [os.path.join(data_dir, ColorSet, x) for x in color_files if is_image_file(x)]
        self.mono_filenames = [os.path.join(data_dir, 'Mono', x) for x in mono_files if is_image_file(x)]
        self.gt_filenames = [os.path.join(data_dir, 'GT', x) for x in gt_files if is_image_file(x)]

        self.patch_size = patch_size
        self.size = len(self.gt_filenames)  # get the size of Gt

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index_ = index % self.size

        color_path = self.color_filenames[index_]
        mono_path = self.mono_filenames[index_]
        gt_path = self.gt_filenames[index_]

        color_path1 = open(color_path, 'rb')
        mono_path1 = open(mono_path, 'rb')
        gt_path1 = open(gt_path, 'rb')
        color_img = Image.open(color_path1)
        mono_img = Image.open(mono_path1)
        gt_img = Image.open(gt_path1)

        color_img = TF.to_tensor(color_img)
        mono_img = TF.to_tensor(mono_img)
        gt_img = TF.to_tensor(gt_img)

        H = color_img.shape[-2]
        W = color_img.shape[-1]

        # Validate on center crop
        if self.patch_size is not None:
            p_H, p_W = self.patch_size
            color_img = TF.center_crop(color_img, (p_H//self.Scale, p_W//self.Scale))
            mono_img = TF.center_crop(mono_img, (p_H, p_W))
            gt_img = TF.center_crop(gt_img, (p_H, p_W))

        # elif self.patch_size is None:
        #     if H % 32 != 0:
        #         new_H = H // 32 * 32
        #     else:
        #         new_H = H
        #     if W % 32 != 0:
        #         new_W = W // 32 * 32
        #     else:
        #         new_W = W
        #     color_img = TF.resize(color_img, (new_H, new_W))
            # mono_img = TF.resize(mono_img, (new_H, new_W))

        filename = os.path.splitext(os.path.split(color_path)[-1])[0]
        color_path1.close()
        mono_path1.close()
        gt_path1.close()

        return color_img, mono_img, gt_img, filename, H, W


class DataLoaderTest(Dataset):
    def __init__(self, data_dir, ColorSet='Color/X4'):
        super().__init__()
        assert os.path.exists(data_dir)
        color_files = sorted(os.listdir(os.path.join(data_dir, ColorSet)))
        mono_files = sorted(os.listdir(os.path.join(data_dir, 'Mono')))

        self.color_filenames = [os.path.join(data_dir, ColorSet, x) for x in color_files if is_image_file(x)]
        self.mono_filenames = [os.path.join(data_dir, 'Mono', x) for x in mono_files if is_image_file(x)]

        self.size = len(self.color_filenames)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        color_path = self.color_filenames[index]
        mono_path = self.mono_filenames[index]

        # color_filename = os.path.splitext(os.path.split(color_path)[-1])
        # color_fileformat = os.path.splitext(os.path.split(color_path)[-1])[-1]
        # mono_filename = os.path.splitext(os.path.split(mono_path)[-1])[0]
        # mono_fileformat = os.path.splitext(os.path.split(mono_path)[-1])[-1]

        color_img = Image.open(color_path)
        mono_img = Image.open(mono_path)
        color_img = TF.to_tensor(color_img)
        mono_img = TF.to_tensor(mono_img)

        H = color_img.shape[-2]
        W = color_img.shape[-1]
        # if H % 32 != 0:
        #     new_H = H // 32 * 32
        # else:
        #     new_H = H
        # if W % 32 != 0:
        #     new_W = W // 32 * 32
        # else:
        #     new_W = W

        color_filename = os.path.splitext(os.path.split(color_path)[-1])
        # color_img = TF.resize(color_img, (new_H, new_W))
        # mono_img = TF.resize(mono_img, (new_H, new_W))

        return color_img, mono_img, color_filename, H, W
