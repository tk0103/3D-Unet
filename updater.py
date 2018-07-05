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
        batchsize = len(predict)
        eps = 1e-16
        loss = -F.mean(F.log(predict+eps) * ground_truth)
        dice = self.jaccard_index(predict,ground_truth)

        chainer.report({"loss":loss}, unet)#mistery
        chainer.report({"dice":dice}, unet)
        return loss

    def jaccard_index(self,predict, ground_truth):
        JI_numerator=0.0
        JI_denominator=0.0

        predict = F.flatten(predict[:,1:4,:,:,:]).data
        ground_truth = F.flatten(ground_truth[:,1:4,:,:,:]).data
        seg = (predict > 0.5)
        #print(aa.shape)

        JI_numerator = (seg * ground_truth).sum()
        JI_denominator =((seg + ground_truth)> 0).sum()

        return JI_numerator/JI_denominator

    def dice_coefficent(self,predict, ground_truth):
        dice_numerator=0.0
        dice_denominator=0.0

        predict = F.flatten(predict).data
        ground_truth = F.flatten(ground_truth).data
        seg = (predict > 0.5)

        dice_numerator = 2*(seg * ground_truth).sum()
        dice_denominator =seg.sum()+ ground_truth.sum()

        return dice_numerator/dice_denominator

    def update_core(self):
        #load optimizer called "unet"
        unet_optimizer = self.get_optimizer("unet")
        batch = self.get_iterator("main").next()#iterator

        # iterator
        label, data = self.converter(batch, self.device)
        unet = self.unet

        predict = unet(data)

        unet_optimizer.update(self.loss_softmax_cross_entropy, unet, predict, label)
