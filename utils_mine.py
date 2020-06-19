import torch
import os
from model import R3Net

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x;

def load_part_of_model(new_model, src_model_path, device_id=0):
    src_model = torch.load(src_model_path, map_location='cuda:' + str(device_id))
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print(k)
        param = src_model.get(k)
        m_dict[k].data = param

    new_model.load_state_dict(m_dict)
    return new_model

if __name__ == '__main__':
    ckpt_path = './ckpt'
    exp_name = 'VideoSaliency_2019-05-14 17:13:16'

    args = {
        'snapshot': '30000',  # your snapshot filename (exclude extension name)
        'crf_refine': False,  # whether to use crf to refine results
        'save_results': True,  # whether to save the resulting masks
        'input_size': (473, 473)
    }
    src_model_path = os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')
    net = R3Net(motion='GRU')
    net = load_part_of_model(net, src_model_path)