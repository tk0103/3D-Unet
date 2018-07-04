#coding:utf-8
import os, sys, time
import argparse, yaml, shutil
import chainer
import chainer.functions as F
import numpy as np
import cupy as cp
from model import UNet3D
from dataset import UnetDataset
import pandas as pd
import util.yaml_utils  as yaml_utils
import util.iomod as io
from chainer import Variable
from chainer.cuda import to_cpu
from chainer.cuda import to_gpu

def main():
    parser = argparse.ArgumentParser(description='Train 3D-Unet')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')
    parser.add_argument('--config_path', type=str, default='configs/base.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='UNet3D_2500.npz',
                        help='Load model data')
    parser.add_argument('--resume', '-res', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--test_list', default='configs/test_list.txt',
                        help='Path to training image list file')

    parser.add_argument('--test_coordinate_list', type=str,
                        default='configs/test_coordinate_list.csv')

    args = parser.parse_args()


    config = yaml_utils.Config(yaml.load(open(os.path.join(os.path.dirname(__file__), args.config_path))))
    print('GPU: {}'.format(args.gpu))
    print('')

    test = UnetDataset(args.root, args.test_list,args.test_coordinate_list, config.patch['patchside'])
    unet = UNet3D(4)

    if(args.gpu>= 0):
        use_cudnn = True
        unet.to_gpu()
    else:
        unet.to_cpu()

    chainer.serializers.load_npz(os.path.join(args.root,args.out,args.model),unet)

    coordi = pd.read_csv(os.path.join(args.root, args.test_coordinate_list),names=("x","y","z")).values.tolist()
    patch_side = 44
    ResultOut = np.zeros((860,544,544),dtype = np.uint8)

    print(len(test))
    for index in range(len(test)):
        t,x = test[index]
        x = x[:,np.newaxis,:]
        x = cp.array(x)

        with chainer.using_config("train", False):
            y = unet(x)
        y = F.softmax(y).data
        pred_label = np.squeeze(to_cpu(y.argmax(axis=1)))
        x,y,z=coordi[index]
        x_s, x_e = (x - int(patch_side/2)), (x + int(patch_side/2))
        y_s, y_e = (y - int(patch_side/2)), (y + int(patch_side/2))
        z_s, z_e = (z - int(patch_side/2)), (z + int(patch_side/2))

        ResultOut[z_s:z_e,y_s:y_e,x_s:x_e] = pred_label.astype(np.uint8)

    io.save_raw(ResultOut, os.path.join(args.root,args.out,"TestResult..raw"),np.uint8)
    print("test done")


if __name__ == '__main__':
    main()
