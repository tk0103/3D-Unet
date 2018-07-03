#coding:utf-8
"""
* @auther tozawa
* @date 2018-6-15
"""
import os, sys, time
import argparse, yaml, shutil
import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

from model import UNet3D
from updater import Unet3DUpdater
from dataset import UnetDataset
import util.yaml_utils  as yaml_utils

def main():
    parser = argparse.ArgumentParser(description='Train 3D-Unet')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')
    parser.add_argument('--config_path', type=str, default='configs/base.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data')
    parser.add_argument('--resume', '-res', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--training_list', default='configs/training_list.txt',
                        help='Path to training image list file')
    parser.add_argument('--validation_list', default='configs/validation_list.txt',
                        help='Path to validation image list file')

    parser.add_argument('--training_coordinate_list', type=str,
                        default='configs/training_coordinate_list.csv')
    parser.add_argument('--validation_coordinate_list', type=str,
                        default='configs/validation_coordinate_list.csv')

    args = parser.parse_args()

    '''
    'https://stackoverflow.com/questions/21005822/what-does-os-path-abspathos-path-joinos-path-dirname-file-os-path-pardir'
    '''
    config = yaml_utils.Config(yaml.load(open(os.path.join(os.path.dirname(__file__), args.config_path))))
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(config.batchsize))
    print('# iteration: {}'.format(config.iteration))
    print('')

    # Load the datasets
    train = UnetDataset(args.root, args.training_list,args.training_coordinate_list, config.patch['patchside'])
    train_iter = chainer.iterators.SerialIterator(train, batch_size=config.batchsize)

    # Set up a neural network to train
    print ('Set up a neural network to train')
    unet = UNet3D(7)
    if args.model:
        chainer.serializers.load_npz(args.model, gen)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        unet.to_gpu()

    #Set up an optimizer
    def make_optimizer(model, alpha=0.00001, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer
    opt_unet = make_optimizer(model = unet,
                            alpha=config.adam['alpha'],
                            beta1=config.adam['beta1'],
                            beta2=config.adam['beta2'])
    #Set up a trainer
    updater = Unet3DUpdater(models=(unet),
                            iterator=train_iter,
                            optimizer={'unet':opt_unet},
                            device=args.gpu)

    def create_result_dir(result_dir, config_path, config):
        """https://github.com/pfnet-research/sngan_projection/blob/master/train.py"""
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        def copy_to_result_dir(fn, result_dir):
            bfn = os.path.basename(fn)
            shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

        copy_to_result_dir(config_path, result_dir)
        copy_to_result_dir(
            config.unet['fn'], result_dir)
        copy_to_result_dir(
            config.updater['fn'], result_dir)

    out = os.path.join(args.root, args.out)
    config_path = os.path.join(os.path.dirname(__file__), args.config_path)
    create_result_dir(out, config_path, config)

    trainer = training.Trainer(updater, (config.iteration, 'iteration'), out=out)

    # Set up logging
    snapshot_interval = (config.snapshot_interval, 'iteration')
    display_interval = (config.display_interval, 'iteration')
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(unet, filename=unet.__class__.__name__ +'_{.updater.iteration}.npz'), trigger=snapshot_interval)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=display_interval))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Print selected entries of the log to stdout
    report_keys = ['epoch', 'iteration', 'unet/loss']
    trainer.extend(extensions.PrintReport(report_keys), trigger=display_interval)

    # Use linear shift
    ext_opt_unet = extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
                                         (config.iteration_decay_start, config.iteration), opt_unet)
    trainer.extend(ext_opt_unet)

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['unet/loss'], 'iteration', file_name='unet_loss.png',trigger=display_interval))

    if args.resume:
        # Resume from a snapshot
        print("Resume training with snapshot:{}".format(args.resume))
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    chainer.config.autotune = True
    print('Start training')
    trainer.run()

if __name__ == '__main__':
    main()
