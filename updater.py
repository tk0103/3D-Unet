#coding:utf-8
"""
@auther tzw
@date 2018-6-15
"""
import os, sys, time
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class Unet3DUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.unet = kwargs.pop("models")
        super(Unet3DUpdater, self).__init__(*args, **kwargs)

    def loss_softmax_cross_entropy(self, unet, predict, ground_truth):
        """
        * @param unet Unet
        * @param predict Output of unet
        * @param ground_truth Ground truth label
        """
        #batchsize,ch,z,y,x
        eps = 1e-16
        cross_entropy = -F.mean(F.log(predict+eps) * ground_truth)

        chainer.report({"loss":cross_entropy}, unet)#mistery
        return loss

    def dice_coefficent(self,unet,predict, ground_truth):
        dice_numerator = 0.0
        dice_denominator = 0.0
        eps = 1e-16
        #print(predict.shape)
        #print(ground_truth.shape)
        predict = F.flatten(predict)
        ground_truth = F.flatten(ground_truth.astype(np.float32))
        #predict = F.flatten(predict[:,1:4,:,:,:])
        #ground_truth = F.flatten(ground_truth[:,1:4,:,:,:].astype(np.float32))

        dice_numerator = F.sum(predict * ground_truth)
        dice_denominator = F.sum(predict + ground_truth)
        dice = 2*dice_numerator/(dice_denominator+eps)
        loss = 1 - dice

        chainer.report({"dice":loss}, unet)
        #print(loss)
        return loss

    def update_core(self):
        #load optimizer called "unet"
        unet_optimizer = self.get_optimizer("unet")
        batch = self.get_iterator("main").next()#iterator

        # iterator
        label, data = self.converter(batch, self.device)
        unet = self.unet

        predict = unet(data)
        #label = label[:,:,20:24,20:24,20:24]
        label = label[:,:,20:72,20:72,20:72]

        unet_optimizer.update(self.dice_coefficent, unet, predict, label)
