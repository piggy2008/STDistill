import os
from PIL import Image

def genrate_davis_test_single_imgs(root, save_path):
    file = open(save_path, 'w')
    folders = os.listdir(root)
    for folder in folders:
        imgs = os.listdir(os.path.join(root, folder))
        imgs.sort()
        for img in imgs:
            name, suffix = os.path.splitext(img)
            file.writelines(os.path.join(folder, name) + '\n')
            print (os.path.join(folder, name))

    file.close()

def genrate_test_single_imgs(root, save_path, dataset='davis_test2'):
    file = open(save_path, 'w')
    folders = os.listdir(os.path.join(root, dataset))
    for folder in folders:
        imgs = os.listdir(os.path.join(root, dataset, folder))
        imgs.sort()
        for img in imgs:
            name, suffix = os.path.splitext(img)
            file.writelines(folder + '/' + name + '\n')
            print (os.path.join(folder, name))

    file.close()

def generate_seq(path, save_path, dataset='davis', batch=6):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()

        for m in range(0, batch - 1):
            image_batch = ''

            for n in range(0, batch - m - 1):
                image = images[0]
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(dataset, folder, name)
                image_batch = image_batch + path_temp + ','

            for a in range(0, m + 1):
                image_ = images[a]
                name_, suffix = os.path.splitext(image_)
                path_temp_ = os.path.join(dataset, folder, name_)
                image_batch = image_batch + path_temp_ + ','
            print(image_batch[:-1])
            file.writelines(image_batch[:-1] + '\n')

        for i in range(0, len(images) - batch + 1):
            image_batch = ''
            for j in range(batch):

                image = images[i + j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(dataset, folder, name)
                if j == (batch - 1):
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_with_step(path, save_path, batch, step):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for i in range(0, len(images) - batch * step + 1):
            image_batch = ''
            for j in range(i, i + batch * step, step):

                image = images[j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join('DAFB2', folder, name)
                if j == (i + batch * step - step):
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_by_MCL(path, gt_path, save_path, batch=5):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()

        gts = os.listdir(os.path.join(gt_path, folder))
        gts.sort()
        for i in range(0, len(gts)):
            image_batch = ''
            image_name, suffix = os.path.splitext(gts[i])
            for j in range(batch):
                image_index = images.index(image_name + '.jpg')
                image = images[image_index - j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 0:
                    image_batch = path_temp + image_batch
                else:
                    image_batch = path_temp + ',' + image_batch
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_by_VOS(path, gt_path, save_path, batch=5):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()

        gts = os.listdir(os.path.join(gt_path, folder))
        gts.sort()
        for i in range(0, len(gts)):
            image_batch = ''
            image_name, suffix = os.path.splitext(gts[i])
            for j in range(batch):
                image_index = images.index(image_name + '.png')
                image = images[image_index - j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 0:
                    image_batch = path_temp + image_batch
                else:
                    image_batch = path_temp + ',' + image_batch
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_seq_by_ViSal(path, gt_path, save_path, batch=5):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()

        gts = os.listdir(os.path.join(gt_path, folder))
        gts.sort()
        for i in range(1, len(gts)):
            image_batch = ''
            image_name, suffix = os.path.splitext(gts[i])
            for j in range(batch):
                image_index = images.index(image_name + '.jpg')
                image = images[image_index - j]
                print (os.path.join(path, folder, image))
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 0:
                    image_batch = path_temp + image_batch
                else:
                    image_batch = path_temp + ',' + image_batch
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def generate_single_by_THUR(path, save_path, batch):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for img in images:
           image_batch = ''
           for i in range(0, batch):
               name, suffix = os.path.splitext(img)
               path_temp = os.path.join('THUR15000', folder, name)
               if i == (batch - 1):
                   image_batch = image_batch + path_temp
               else:
                   image_batch = image_batch + path_temp + ','
           print(image_batch)
           file.writelines(image_batch + '\n')

    file.close()

def generate_single_by_DAVSOD(path, save_path):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    for folder in folders:
        images = os.listdir(os.path.join(path, folder, 'Imgs'))
        images.sort()
        for img in images:
            img_temp = os.path.join('DAVSOD', folder, 'Imgs', img)
            gt_temp = os.path.join('DAVSOD', folder, 'GT_object_level', img[:-4] + '.png')
            print(img_temp + ' ' + gt_temp)
            file.writelines(img_temp + ' ' + gt_temp + '\n')

    file.close()

def generate_seq_by_DUTS_TR(path, save_path):
    folders = os.listdir(path)
    folders.sort()
    file = open(save_path, 'w')
    images = os.listdir(os.path.join(path, 'DUTS-TR-Image'))
    images.sort()
    for img in images:
       name, suffix = os.path.splitext(img)
       img_temp = os.path.join('DUTS-TR', 'DUTS-TR-Image', img)
       gt_temp = os.path.join('DUTS-TR', 'DUTS-TR-Mask', name + '.png')
       print(img_temp + ' ' + gt_temp)
       file.writelines(img_temp + ' ' + gt_temp + '\n')

    file.close()

def generate_seq_by_DUT_OMRON(path, save_path, batch):
    file = open(save_path, 'w')
    images = os.listdir(path)
    images.sort()
    for img in images:
       image_batch = ''
       for i in range(0, batch):
           name, suffix = os.path.splitext(img)
           path_temp = os.path.join('DUT-OMRON', name)
           if i == (batch - 1):
               image_batch = image_batch + path_temp
           else:
               image_batch = image_batch + path_temp + ','
       print(image_batch)
       file.writelines(image_batch + '\n')

    file.close()

def generate_seq_by_MSRA10K(path, save_path, batch):
    file = open(save_path, 'w')
    images = os.listdir(path)
    images.sort()
    for img in images:
       image_batch = ''
       for i in range(0, batch):
           name, suffix = os.path.splitext(img)
           path_temp = os.path.join('MSRA10K_Imgs_GT', name)
           if i == (batch - 1):
               image_batch = image_batch + path_temp
           else:
               image_batch = image_batch + path_temp + ','
       print(image_batch)
       file.writelines(image_batch + '\n')

    file.close()


def remove_folder(path, remove_folder):
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        imgs = os.listdir(os.path.join(path, folder, remove_folder))
        imgs.sort()
        for i, img in enumerate(imgs):
            image = Image.open(os.path.join(path, folder, remove_folder, img)).convert('RGB')
            os.remove(os.path.join(path, folder, remove_folder, img))
            image.save(os.path.join(path, folder, img))
            if i == (len(imgs) - 1):
                os.rmdir(os.path.join(path, folder, remove_folder))

def replace_suffix(path, suffix):
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        imgs = os.listdir(os.path.join(path, folder))
        imgs.sort()
        for i, img in enumerate(imgs):
            image = Image.open(os.path.join(path, folder, img)).convert('RGB')
            image.save(os.path.join(path, folder, img[:-4] + suffix))
            os.remove(os.path.join(path, folder, img))



def change_map_path(path):
    import shutil
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        imgs = os.listdir(os.path.join(path, folder, 'Sal'))
        imgs.sort()
        for i in range(0, len(imgs)):
            shutil.move(os.path.join(path, folder, 'Sal', imgs[i]), os.path.join(path, folder, imgs[i]))
            if i == (len(imgs)-1):
                os.rmdir(os.path.join(path, folder, 'Sal'))
            # os.rename(os.path.join(path, folder, img))

def generate_single_from_seq(root_path, save_path):
    lines = open(root_path)
    lines2 = open(save_path, 'w')
    for i, line in enumerate(lines):
        print(line.strip())
        names = line.strip().split(',')
        for i, name in enumerate(names):
            if i > 1:
                folder, img_name = name.split('/')
                lines2.writelines('/home/ty/data/VOS_test/' + folder + '/Imgs/' + img_name + '.png 0\n')

    lines.close()
    lines2.close()

def remove_pre(root):
    folders = os.listdir(root)
    folders.sort()
    for folder in folders:
        imgs = os.listdir(os.path.join(root, folder))
        for img in imgs:
            new_name = img.replace(folder + '_', '')
            os.rename(os.path.join(root, folder, img), os.path.join(root, folder, new_name))

def generate_davis_trainingset(root, test_root, save_path):
    all_folders = os.listdir(root)
    all_folders.sort()
    test_folders = os.listdir(test_root)
    test_folders.sort()
    save_folders = open(save_path, 'w')
    # for folder in all_folders:
    #     if folder not in test_folders:
    #         save_folders.writelines(folder + '\n')
    #         print('save_folder:', folder)
    for folder in test_folders:
        save_folders.writelines(folder + '\n')
        print('save_folder:', folder)

    save_folders.close()



if __name__ == '__main__':
    root = '/home/ty/data/FBMS/FBMS_Testset2'
    gt_root = '/home/ty/data/FBMS/GT'
    save_path = '/home/ty/data/FBMS/FBMS_test_single.txt'
    # root = '/home/qub/data/saliency/SegTrack-V2/SegTrackV2_test'
    # gt_root = '/home/qub/data/saliency/VOS/GT'
    # save_path = '/home/qub/data/saliency/SegTrack-V2/SegTrackV2_test_5f.txt'

    # save_path = '/home/ty/data/Pre-train/pretrain_all_DAVSOD.txt'
    # genrate_davis_test_single_imgs(root, save_path)
    # generate_seq_by_DUTS_TR(root, save_path)
    # generate_seq_by_MCL(root, gt_root, save_path)
    # generate_single_by_DAVSOD(root, save_path)
    # generate_seq(root, save_path, dataset='')
    genrate_test_single_imgs(gt_root, save_path, dataset='')
    # generate_seq_by_VOS(root, gt_root, save_path, batch=6)
    # generate_seq_by_ViSal(root, gt_root, save_path)
    # replace_suffix('/home/ty/data/video_saliency/train_all/DAVSOD', '.jpg')
    # remove_folder('/home/ty/data/video_saliency/train_all/DAVSOD', 'Imgs')
    # import random
    # num_frames = 100
    # clip_length = 5
    # x = 4  # 6
    # k = random.randint(1, x)
    # print('k = ', k)
    # top = num_frames - clip_length * k - 1
    # while top <= 1:
    #     x -= 1
    #     k = random.randint(1, x)
    #     top = num_frames - clip_length * k - 1
    #
    # rand_frame = random.randint(0, top)
    # frames = []
    # labels = []
    # atts = []
    #
    # for i in range(rand_frame, rand_frame + k *clip_length, k):
    #     print('i = ', i)

    # root = '/home/ty/data/davis/480p'
    # test_root = '/home/ty/data/davis/davis_test2'
    # save_path = '/home/ty/data/davis/davis_val_seqname.txt'
    # generate_davis_trainingset(root, test_root, save_path)


