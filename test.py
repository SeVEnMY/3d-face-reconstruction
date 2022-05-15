import os
import argparse
from models import create_model
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from scipy.io import loadmat, savemat

def initialize_opt(parser):
    
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--dataset_mode', type=str, default=None)
    parser.add_argument('--img_folder', type=str, default='examples')
    parser.add_argument('--name', type=str, default='twoloss')
    parser.add_argument('--model', type=str, default='facerecon')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--bfm_folder', type=str, default='BFM')
    parser.add_argument('--epoch', type=str, default=20)
    parser.isTrain = False
    opt, _ = parser.parse_known_args()
    return opt

def data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def load_data(im_path, lm_path, lm3d_std, to_tensor=True):
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm

def main(rank, opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()

    im_path, lm_path = data_path(name)
    lm3d_std = load_lm3d(opt.bfm_folder) 

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        if not os.path.isfile(lm_path[i]):
            continue
        im_tensor, lm_tensor = load_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor
        }
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        model.save_mesh(os.path.join('./checkpoints/twoloss/results', name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        model.save_coeff(os.path.join('./checkpoints/twoloss/results', name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process balabla.')
    opt = initialize_opt(parser)
    main(0, opt, 'examples')
    
