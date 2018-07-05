#coding:utf-8
"""
@auther tk0103
@date 2018-07-04
"""

from chainer.training import extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
import numpy as np
from chainer import reporter as reporter_module
from chainer.training import extension
import chainer.functions as F

class UNet3DEvaluator(extensions.Evaluator):
    def __init__(self, iterator, unet,eval_func=None,converter=convert.concat_examples,device=None, eval_hook=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self._targets = {'unet':unet}
        self._eval_func = eval_func
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    def jaccard_index(self,predict, ground_truth):
        JI_numerator=0.0
        JI_denominator=0.0

        predict = F.flatten(predict[:,1:4,:,:,:]).data
        ground_truth = F.flatten(ground_truth[:,1:4,:,:,:]).data
        seg = (predict > 0.5)

        JI_numerator = (seg * ground_truth).sum()
        JI_denominator =((seg + ground_truth)> 0).sum()

        if(JI_denominator == 0.0):
            JI_denominator = 1e-10
        return JI_numerator/JI_denominator

    def loss_softmax_cross_entropy(self, unet, predict, ground_truth):
        eps = 1e-16
        cross_entropy = -F.mean(F.log(predict+eps) * ground_truth)

        chainer.report({"loss":cross_entropy}, unet)#mistery
        return loss

    def dice_coefficent(self,predict, ground_truth):
        dice_numerator = 0.0
        dice_denominator = 0.0
        eps = 1e-16

        predict = F.flatten(predict[:,1:4,:,:,:])
        ground_truth = F.flatten(ground_truth[:,1:4,:,:,:].astype(np.float32))

        dice_numerator = F.sum(predict * ground_truth)
        dice_denominator =F.sum(predict+ ground_truth)
        dice = 2*dice_numerator/(dice_denominator+eps)

        return dice

    def evaluate(self):
        iterator = self._iterators['main']
        unet = self._targets['unet']
        #eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                ground_truth,data = self.converter(batch, self.device)
                predict = unet(data)
                #observation['vali/unet/loss'] = loss
                observation['vali/unet/dice'] = self.dice_coefficent(predict,ground_truth)
            summary.add(observation)

        return summary.compute_mean()
