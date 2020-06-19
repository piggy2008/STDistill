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

class DAVIS2016(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=(473, 473),
                 db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2016',
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
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        if self.train:
            fname = 'davis_train_seqname'
        else:
            fname = 'val_seqs'

        if self.seq_name is None:

            # Initialize the original DAVIS splits for training the parent network
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                video_list = []
                labels = []
                image_list = []
                img_labels = []
                seq_len = {}
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, '480p/', seq.strip())))
                    images_path = list(map(lambda x: os.path.join('480p/', seq.strip(), x), images))
                    strat_num = len(video_list)
                    video_list.extend(images_path)
                    end_num = len(video_list)
                    seq_len[seq.strip()] = np.array([strat_num, end_num])
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'GT/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('GT/', seq.strip(), x), lab))
                    labels.extend(lab_path)

                images_name = [i_id.strip() for i_id in open(imgs_file)]
                images_name.sort()
                for names in images_name:
                    img_name, gt_name = names.split(' ')
                    image_list.append(os.path.join(db_image_dir, img_name))
                    img_labels.append(os.path.join(db_image_dir, gt_name))

        else:

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, '480p/', str(seq_name))))
            video_list = list(map(lambda x: os.path.join('480p/', str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'GT/', str(seq_name))))
            labels = [os.path.join('GT/', str(seq_name), name_label[0])]
            labels.extend([None]*(len(names_img)-1))
            if self.train:
                img_list = [video_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(video_list))

        self.image_list = image_list
        self.img_labels = img_labels
        self.video_list = video_list
        self.labels = labels
        self.seq_len = seq_len

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        print(len(self.video_list), len(self.image_list))
        return len(self.video_list)

    def __getitem__(self, idx):
        video, video_gt = self.make_video_gt_pair(idx)
        video_id = idx
        img_idx = np.random.randint(1, len(self.image_list) - 1)
        seq_name = self.video_list[idx].split('/')[-2]
        if self.train:
            [start, end] = self.seq_len[seq_name]
            search_id = start
            # search_id = np.random.randint(start, end)
            # if search_id == video_id:
            #     search_id = np.random.randint(start, end)
            search, search_gt = self.make_video_gt_pair(search_id)
            img, img_gt = self.make_img_gt_pair(img_idx)
            sample = {'video': video, 'video_gt': video_gt, 'search': search, 'search_gt': search_gt, 'img': img, 'img_gt': img_gt}

            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

            if self.transform is not None:
                sample = self.transform(sample)
        else:
            video, video_gt = self.make_video_gt_pair(idx)
            sample = {'video': video, 'video_gt': video_gt}
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.image_list[idx]), cv2.IMREAD_COLOR)
        if self.img_labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.img_labels[idx]), cv2.IMREAD_GRAYSCALE)
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

    def make_video_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.video_list[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
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
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        img = img.transpose((2, 0, 1))
        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)
            gt[gt != 0] = 1

        # seq_name = self.video_list[idx].split('/')[2]
        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt
    from dataloader.helpers import overlay_mask, im_normalize, tens2image

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    dataset = DAVIS2016(db_root_dir='/home/ty/data/davis',
                        train=True, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        # plt.figure()
        # plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
        # if i == 10:
        #     break
        print(data['img'].size())
        print(data['img_gt'].size())

    # plt.show(block=True)