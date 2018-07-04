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

    parser.add_argument('--validation_list', default='configs/validation_list.txt',
                        help='Path to training image list file')

    parser.add_argument('--valida_coordinate_list', type=str,
                        default='configs/validation_coordinate_list.csv')

    args = parser.parse_args()


    config = yaml_utils.Config(yaml.load(open(os.path.join(os.path.dirname(__file__), args.config_path))))
    print('GPU: {}'.format(args.gpu))
    print('')

    validation = UnetDataset(args.root, args.validation_list,args.valida_coordinate_list, config.patch['patchside'])
    unet = UNet3D(4)

    if(args.gpu>= 0):
        use_cudnn = True
        unet.to_gpu()
    else:
        unet.to_cpu()

    chainer.serializers.load_npz(os.path.join(args.root,args.out,args.model),unet)
    eps = 1e-16
    loss = 0.0
    for index in range(len(validation)):
        t,x = validation[index]
        x = x[:,np.newaxis,:]
        x = cp.array(x)
        t = cp.array(t)
        with chainer.using_config("train", False):
            y = unet(x).data
        y = cp.squeeze(y)
        loss += to_cpu(cp.log(y+eps) * t)
    print(loss/len(validation))



if __name__ == '__main__':
    main()
