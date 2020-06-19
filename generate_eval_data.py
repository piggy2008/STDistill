import os
from PIL import Image

ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2019-08-15 05:22:35'
name = 'davis'
root = '/home/qub/data/saliency/davis/davis_test2'
gt_root = '/home/qub/data/saliency/davis/GT'

args = {
    'snapshot': '20000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True  # whether to save the resulting masks
}

save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']))

new_save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s_eval' % (exp_name, name, args['snapshot']))
folders = os.listdir(save_path)
folders.sort()

for folder in folders:
    imgs = os.listdir(os.path.join(save_path, folder))
    imgs.sort()

    for img in imgs:
        print(os.path.join(folder, img))
        pred = Image.open(os.path.join(save_path, folder, img)).convert('L')
        if not os.path.exists(os.path.join(new_save_path, folder, '1')):
            os.makedirs(os.path.join(new_save_path, folder, '1'))

        pred.save(os.path.join(new_save_path, folder, '1', img))