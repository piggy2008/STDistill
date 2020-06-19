import os
import numpy as np
import cv2
from scipy.misc import imresize


from torch.utils.data import Dataset
import random

def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def scale_gt(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST).astype(float)


def image_crop(img, gt):
    H = int(0.9 * img.shape[0])
    W = int(0.9 * img.shape[1])
    H_offset = random.choice(range(img.shape[0] - H))
    W_offset = random.choice(range(img.shape[1] - W))
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)
    img = img[H_slice, W_slice, :]
    gt = gt[H_slice, W_slice]

    return img, gt

class DAVIS_Single(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=(473, 473),
                 db_image_dir='/home/ty/data/Pre-train',
                 imgs_file='/home/ty/data/Pre-train/pretrain_all_seq_DUT_TR_DAFB2_DAVSOD.txt',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.inputRes = inputRes
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name


        image_list = []
        img_labels = []
        for i_id in open(imgs_file):
            line = i_id.strip()
            image_name, label_name = line.split(' ')
            image_list.append(os.path.join(db_image_dir, image_name))
            img_labels.append(os.path.join(db_image_dir, label_name))




        assert (len(img_labels) == len(image_list))

        self.image_list = image_list
        self.img_labels = img_labels


        print('Done initializing training Dataset')

    def __len__(self):
        print(len(self.image_list))
        return len(self.image_list)

    def __getitem__(self, idx):


        img, img_gt = self.make_img_gt_pair(idx)
        sample = {'img': img, 'img_gt': img_gt}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.image_list[idx]), cv2.IMREAD_COLOR)
        if self.img_labels[idx] is not None:
            label = cv2.imread(os.path.join(self.img_labels[idx]), cv2.IMREAD_GRAYSCALE)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.train:
            img, label = image_crop(img, label)
            scale = random.uniform(0.7, 1.3)
            flip_pro = random.uniform(0, 1)
            img_tmp = scale_im(img, scale)
            img_tmp = flip(img_tmp, flip_pro)
            gt_tmp = scale_gt(label, scale)
            gt_tmp = flip(gt_tmp, flip_pro)

            img = img_tmp
            label = gt_tmp

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            if self.img_labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        img = img.transpose((2, 0, 1))
        if self.img_labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)
            gt[gt != 0] = 1


        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.image_list[0]))

        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt
    from dataloader.helpers import overlay_mask, im_normalize, tens2image

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    dataset = DAVIS_Single(train=True, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        # plt.figure()
        # plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
        # if i == 10:
        #     break
        print(data['img'].size())
        print(data['img_gt'].size())

    # plt.show(block=True)