import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from matplotlib import pyplot as plt
import random
import torchvision
import numpy as np
from joint_transforms import crop, scale, flip


def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
    return [(os.path.join(root, img_name + '.jpg'), os.path.join(root, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class VideoImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, imgs_file, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = [i_id.strip() for i_id in open(imgs_file)]
        self.imgs.sort()
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index].split(' ')
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        target = Image.open(os.path.join(self.root, gt_path)).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class VideoImage2Folder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, imgs_file, video_root, video_gt_root, joint_transform=None,
                 joint_seq_transform=None, input_size=(473, 473), transform=None, target_transform=None):
        self.root = root
        self.imgs = [i_id.strip() for i_id in open(imgs_file)]
        self.imgs.sort()
        self.joint_transform = joint_transform
        self.joint_seq_transform = joint_seq_transform
        self.transform = transform
        self.target_transform = target_transform
        self.video_root = video_root
        self.video_gt_root = video_gt_root
        self.input_size = input_size

        video_list = []
        video_gt_list = []
        video_num = {}
        folders = os.listdir(video_root)
        for folder in folders:
            video_names = os.listdir(os.path.join(video_root, folder))
            video_names.sort()
            video_path = list(map(lambda x: os.path.join(video_root, folder, x), video_names))
            start_num = len(video_list)
            video_list.extend(video_path)
            end_num = len(video_list)
            video_num[folder] = np.array([start_num, end_num])

            video_gt_names = os.listdir(os.path.join(video_gt_root, folder))
            video_gt_names.sort()
            video_gt_path = list(map(lambda x: os.path.join(video_gt_root, folder, x), video_gt_names))
            video_gt_list.extend(video_gt_path)

        self.video_list = video_list
        self.video_gt_list = video_gt_list
        self.video_num = video_num


    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index].split(' ')
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        target = Image.open(os.path.join(self.root, gt_path)).convert('L')

        previous_frame, previous_gt, current_frame, \
        current_gt, next_frame, next_gt = self.generate_video_seq2()

        if self.joint_seq_transform is not None:
            previous_frame, previous_gt = self.joint_seq_transform(previous_frame, previous_gt)
            current_frame, current_gt = self.joint_seq_transform(current_frame, current_gt)
            next_frame, next_gt = self.joint_seq_transform(next_frame, next_gt)


        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)
            previous_frame = self.transform(previous_frame)
            current_frame = self.transform(current_frame)
            next_frame = self.transform(next_frame)
        if self.target_transform is not None:
            target = self.target_transform(target)
            previous_gt = self.target_transform(previous_gt)
            current_gt = self.target_transform(current_gt)
            next_gt = self.target_transform(next_gt)

        return img, target, previous_frame, previous_gt, current_frame, current_gt, next_frame, next_gt

    def generate_video_seq2(self):
        video_index = random.randint(1, len(self.video_list) - 2)
        # video_index = len(self.video_list) - 1
        seq_name = self.video_list[video_index].split('/')[-2]
        [start, end] = self.video_num[seq_name]

        step = 5

        if video_index <= start + step:
            current_frame = Image.open(self.video_list[video_index + 1]).convert('RGB')
            previous_frame = Image.open(self.video_list[video_index]).convert('RGB')
            next_frame = Image.open(self.video_list[video_index + 2]).convert('RGB')

            current_gt = Image.open(self.video_gt_list[video_index + 1]).convert('L')
            previous_gt = Image.open(self.video_gt_list[video_index]).convert('L')
            next_gt = Image.open(self.video_gt_list[video_index + 2]).convert('L')
        elif video_index >= end - step:
            current_frame = Image.open(self.video_list[video_index - 1]).convert('RGB')
            previous_frame = Image.open(self.video_list[video_index - 2]).convert('RGB')
            next_frame = Image.open(self.video_list[video_index]).convert('RGB')

            current_gt = Image.open(self.video_gt_list[video_index - 1]).convert('L')
            previous_gt = Image.open(self.video_gt_list[video_index - 2]).convert('L')
            next_gt = Image.open(self.video_gt_list[video_index]).convert('L')
        else:
            span = random.randint(1, step)
            current_frame = Image.open(self.video_list[video_index]).convert('RGB')
            previous_frame = Image.open(self.video_list[video_index - span]).convert('RGB')
            next_frame = Image.open(self.video_list[video_index + span]).convert('RGB')

            current_gt = Image.open(self.video_gt_list[video_index]).convert('L')
            previous_gt = Image.open(self.video_gt_list[video_index - span]).convert('L')
            next_gt = Image.open(self.video_gt_list[video_index + span]).convert('L')

        w, h = current_frame.size
        tw = int(0.9 * w)
        th = int(0.9 * h)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        scale_num = random.uniform(0.7, 1.3)
        flip_p = random.uniform(0, 1)

        previous_frame, previous_gt = self.image_aug(previous_frame, previous_gt, scale_num, flip_p, tw, th, x1, y1)
        current_frame, current_gt = self.image_aug(current_frame, current_gt, scale_num, flip_p, tw, th, x1, y1)
        next_frame, next_gt = self.image_aug(next_frame, next_gt, scale_num, flip_p, tw, th, x1, y1)

        if self.input_size is not None:
            previous_frame, previous_gt = previous_frame.resize(self.input_size, Image.BILINEAR), previous_gt.resize(self.input_size, Image.NEAREST)
            current_frame, current_gt = current_frame.resize(self.input_size, Image.BILINEAR), current_gt.resize(self.input_size, Image.NEAREST)
            next_frame, next_gt = next_frame.resize(self.input_size, Image.BILINEAR), next_gt.resize(self.input_size, Image.NEAREST)

        return previous_frame, previous_gt, current_frame, current_gt, next_frame, next_gt

    def image_aug(self, img, gt, scale_num, flip_p, tw, th, x1, y1):
        img, gt = crop(img, gt, tw, th, x1, y1)
        img, gt = scale(img, gt, scale_num)
        img, gt = flip(img, gt, flip_p)

        return img, gt

    def generate_video_seq(self):
        video_index = random.randint(1, len(self.video_list) - 2)
        # video_index = len(self.video_list) - 1
        seq_name = self.video_list[video_index].split('/')[-2]
        [start, end] = self.video_num[seq_name]

        if video_index == start:
            current_frame = Image.open(self.video_list[video_index + 1]).convert('RGB')
            previous_frame = Image.open(self.video_list[video_index]).convert('RGB')
            next_frame = Image.open(self.video_list[video_index + 2]).convert('RGB')

            current_gt = Image.open(self.video_gt_list[video_index + 1]).convert('L')
            previous_gt = Image.open(self.video_gt_list[video_index]).convert('L')
            next_gt = Image.open(self.video_gt_list[video_index + 2]).convert('L')
        elif video_index == end:
            current_frame = Image.open(self.video_list[video_index - 1]).convert('RGB')
            previous_frame = Image.open(self.video_list[video_index - 2]).convert('RGB')
            next_frame = Image.open(self.video_list[video_index]).convert('RGB')

            current_gt = Image.open(self.video_gt_list[video_index - 1]).convert('L')
            previous_gt = Image.open(self.video_gt_list[video_index - 2]).convert('L')
            next_gt = Image.open(self.video_gt_list[video_index]).convert('L')
        else:
            current_frame = Image.open(self.video_list[video_index]).convert('RGB')
            previous_frame = Image.open(self.video_list[video_index - 1]).convert('RGB')
            next_frame = Image.open(self.video_list[video_index + 1]).convert('RGB')

            current_gt = Image.open(self.video_gt_list[video_index]).convert('L')
            previous_gt = Image.open(self.video_gt_list[video_index - 1]).convert('L')
            next_gt = Image.open(self.video_gt_list[video_index + 1]).convert('L')

        return previous_frame, previous_gt, current_frame, current_gt, next_frame, next_gt

    def __len__(self):
        print('video length:', len(self.video_list), '----', 'image length:', len(self.imgs))
        return len(self.imgs)


class VideoSequenceFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, gt_root, imgs_file, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.gt_root = gt_root
        self.imgs = [i_id.strip() for i_id in open(imgs_file)]
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_paths = self.imgs[index].split(',')
        img_list = []
        gt_list = []
        for img_path in img_paths:
            img = Image.open(os.path.join(self.root, img_path + '.jpg')).convert('RGB')
            target = Image.open(os.path.join(self.gt_root, img_path + '.png')).convert('L')
            img_list.append(img)
            gt_list.append(target)
        if self.joint_transform is not None:
            img_list, gt_list = self.joint_transform(img_list, gt_list)
        if self.transform is not None:
            imgs = []
            for img_s in img_list:
                imgs.append(self.transform(img_s).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
        if self.target_transform is not None:
            targets = []
            for target_s in gt_list:
                targets.append(self.target_transform(target_s).unsqueeze(0))
            targets = torch.cat(targets, dim=0)
        return imgs, targets

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    from torchvision import transforms

    import joint_transforms
    from torch.utils.data import DataLoader
    from config import msra10k_path, video_seq_path, video_seq_gt_path, video_train_path
    import numpy as np
    joint_transform = joint_transforms.Compose([
        joint_transforms.ImageResize(550),
        joint_transforms.RandomCrop(473),
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomRotate(10)
    ])

    joint_seq_transform = joint_transforms.Compose([
        joint_transforms.ImageResize(520),
        joint_transforms.RandomCrop(473)
    ])

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()
    input_size = (473, 473)
    # imgs_file = '/home/ty/data/video_saliency/train_all_DAFB2_DAVSOD_5f.txt'
    # train_set = VideoSequenceFolder(video_seq_path, video_seq_gt_path, imgs_file, joint_transform, img_transform, target_transform)
    imgs_file = '/home/ty/data/Pre-train/pretrain_all_seq_DUT_TR_DAFB2_DAVSOD.txt'
    # train_set = VideoImageFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)
    video_root = '/home/ty/data/video_saliency/train_all/DAFB2_DAVSOD'
    video_gt_root = '/home/ty/data/video_saliency/train_all_gt2_revised/DAFB2_DAVSOD'

    train_set = VideoImage2Folder(video_train_path, imgs_file, video_root, video_gt_root, joint_transform, None, input_size, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=6, num_workers=12, shuffle=False)

    for i, data in enumerate(train_loader):
        input, target, previous_frame, previous_gt, current_frame, current_gt, next_frame, next_gt = data
        input = current_gt.squeeze(0)
        target = previous_gt.squeeze(0)
        input = input.data.cpu().numpy()
        target = target.data.cpu().numpy()
        # np.savetxt('image.txt', input[0, 0, :, :])
        input = input.transpose(0, 2, 3, 1)
        target = target.transpose(0, 2, 3, 1)
        # # for i in range(0, input.shape[0]):
        # plt.subplot(2, 2, 1)
        # plt.imshow(input[0, :, :, 0])
        # plt.subplot(2, 2, 2)
        # plt.imshow(target[0, :, :, 0])
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(input[1, :, :, 0])
        # plt.subplot(2, 2, 4)
        # plt.imshow(target[1, :, :, 0])
        # #
        # plt.show()
        print(input.shape)