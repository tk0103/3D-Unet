import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import HeNormal as w

class UNet3D(chainer.Chain):

    def __init__(self, label):
        super(UNet3D, self).__init__()
        with self.init_scope():
            #encorder pass
            self.conv1 = L.ConvolutionND(ndim=3,in_channels=1,out_channels=8, ksize=3,pad=0)
            self.bnc0 = L.BatchNormalization(8)
            self.conv2 = L.ConvolutionND(ndim=3,in_channels=8,out_channels=16, ksize=3,pad=0)
            self.bnc1 = L.BatchNormalization(16)

            self.conv3 = L.ConvolutionND(ndim=3,in_channels=16,out_channels=16, ksize=3,pad=0)
            self.bnc2 = L.BatchNormalization(16)
            self.conv4 = L.ConvolutionND(ndim=3,in_channels=16,out_channels=32, ksize=3,pad=0)
            self.bnc3 = L.BatchNormalization(32)

            self.conv5 = L.ConvolutionND(ndim=3,in_channels=32,out_channels=32, ksize=3,pad=0)
            self.bnc4 = L.BatchNormalization(32)
            self.conv6 = L.ConvolutionND(ndim=3,in_channels=32,out_channels=64, ksize=3,pad=0)
            self.bnc5 = L.BatchNormalization(64)

            #decorder pass
            self.dconv1 = L.DeconvolutionND(ndim=3,in_channels=64,out_channels=64, ksize=2, stride=2)
            self.conv7 = L.ConvolutionND(ndim=3,in_channels=32 + 64,out_channels=32, ksize=3,pad=0)
            self.bnd4 = L.BatchNormalization(32)
            self.conv8 = L.ConvolutionND(ndim=3,in_channels=32,out_channels=32, ksize=3,pad=0)
            self.bnd3 = L.BatchNormalization(32)

            self.dconv2 = L.DeconvolutionND(ndim=3,in_channels=32,out_channels=32, ksize=2, stride=2)
            self.conv9 = L.ConvolutionND(ndim=3,in_channels=16 + 32,out_channels=16, ksize=3,pad=0)
            self.bnd2 = L.BatchNormalization(16)
            self.conv10 = L.ConvolutionND(ndim=3,in_channels=16,out_channels=16, ksize=3,pad=0)
            self.bnd1 = L.BatchNormalization(16)
            self.lcl = L.ConvolutionND(ndim=3, in_channels=16, out_channels=label, ksize=1, pad=0)

    def __call__(self, x):
        h1 = F.relu(self.bnc0(self.conv1(x)))
        h2 = F.relu(self.bnc1(self.conv2(h1)))
        h3 = F.max_pooling_nd(h2, ksize=2, stride=2)

        h4 = F.relu(self.bnc2(self.conv3(h3)))
        h5 = F.relu(self.bnc3(self.conv4(h4)))
        h6 = F.max_pooling_nd(h5,ksize=2,stride=2)

        h7 = F.relu(self.bnc4(self.conv5(h6)))
        h8 = F.relu(self.bnc5(self.conv6(h7)))

        h9 = self.dconv1(h8)

        h10 = F.concat([h9, self.cropping(h5,h9)])
        h11 = F.relu(self.bnd4(self.conv7(h10)))
        h12 = F.relu(self.bnd3(self.conv8(h11)))
        h13 = self.dconv2(h12)

        h14 = F.concat([h13, self.cropping(h2,h13)])
        h15 = F.relu(self.bnd2(self.conv9(h14)))
        h16 = F.relu(self.bnd1(self.conv10(h15)))
        lcl = F.softmax(self.lcl(h16), axis=1)
        #print(x.shape)
        #print(h16.shape)

        return lcl


    def cropping(self, input, ref):

        edgez = (input.shape[2] - ref.shape[2])/2
        edgey = (input.shape[3] - ref.shape[3])/2
        edgex = (input.shape[4] - ref.shape[4])/2
        edgez = int(edgex)
        edgey = int(edgey)
        edgex = int(edgez)

        X = F.split_axis(input,(edgex,int(input.shape[4]-edgex)),axis=4)
        X = X[1]
        X = F.split_axis(X,(edgey,int(X.shape[3]-edgey)),axis=3)
        X = X[1]
        X = F.split_axis(X,(edgez,int (X.shape[2]-edgez)),axis=2)
        X=X[1]
        return X
