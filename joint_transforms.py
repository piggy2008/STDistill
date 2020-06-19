import numbers
import random

from PIL import Image, ImageOps

def flip(img, mask, flip_p):
    if flip_p < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def scale(img, mask, ratio):
    new_dims = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    return img.resize(new_dims, Image.BILINEAR), mask.resize(new_dims, Image.NEAREST)

def crop(img, mask, tw, th, x1, y1):

    return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        # assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ImageResize(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        if isinstance(img, list) and isinstance(mask, list):
            img_resized = []
            mask_resized = []
            for img_s, mask_s in zip(img, mask):
                tw, th = self.size
                img_resized.append(img_s.resize((tw, th), Image.BILINEAR))
                mask_resized.append(mask_s.resize((tw, th), Image.BILINEAR))
            return img_resized, mask_resized
        else:
            tw, th = self.size
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if isinstance(img, list) and isinstance(mask, list):
            imgs_crop = []
            masks_crop = []
            img_first = True
            for img_s, mask_s in zip(img, mask):
                if self.padding > 0:
                    img_s = ImageOps.expand(img_s, border=self.padding, fill=0)
                    mask_s = ImageOps.expand(mask_s, border=self.padding, fill=0)

                assert img_s.size == mask_s.size
                w, h = img_s.size
                th, tw = self.size

                if w < tw or h < th:
                    imgs_crop.append(img_s.resize((tw, th), Image.BILINEAR))
                    masks_crop.append(mask_s.resize((tw, th), Image.NEAREST))
                else:
                    if img_first:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        img_first = False
                    imgs_crop.append(img_s.crop((x1, y1, x1 + tw, y1 + th)))
                    masks_crop.append(mask_s.crop((x1, y1, x1 + tw, y1 + th)))
            return imgs_crop, masks_crop
        else:
            if self.padding > 0:
                img = ImageOps.expand(img, border=self.padding, fill=0)
                mask = ImageOps.expand(mask, border=self.padding, fill=0)

            assert img.size == mask.size
            w, h = img.size
            th, tw = self.size
            if w == tw and h == th:
                return img, mask
            if w < tw or h < th:
                return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if isinstance(img, list) & isinstance(mask, list):
            if random.random() < 0.5:
                img_flips = []
                mask_flips = []
                for img_s, mask_s in zip(img, mask):
                    img_flips.append(img_s.transpose(Image.FLIP_LEFT_RIGHT))
                    mask_flips.append(mask_s.transpose(Image.FLIP_LEFT_RIGHT))
                return img_flips, mask_flips
            return img, mask
        else:
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        if isinstance(img, list) & isinstance(mask, list):
            rotate_degree = random.random() * 2 * self.degree - self.degree
            img_rotates = []
            mask_rotates = []
            for img_s, mask_s in zip(img, mask):
                img_rotates.append(img_s.rotate(rotate_degree, Image.BILINEAR))
                mask_rotates.append(mask_s.rotate(rotate_degree, Image.NEAREST))
            return img_rotates, mask_rotates
        else:
            rotate_degree = random.random() * 2 * self.degree - self.degree
            return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)
