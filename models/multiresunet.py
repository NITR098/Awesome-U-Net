# https://github.com/j-sripad/mulitresunet-pytorch/blob/main/multiresunet.py

from typing import Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
import torch


class Multiresblock(nn.Module):
  def __init__(self,input_features : int, corresponding_unet_filters : int ,alpha : float =1.67)->None:
    """
        MultiResblock
        Arguments:
          x - input layer
          corresponding_unet_filters - Unet filters for the same stage
          alpha - 1.67 - factor used in the paper to dervie number of filters for multiresunet filters from Unet filters
        Returns - None
    """ 
    super().__init__()
    self.corresponding_unet_filters = corresponding_unet_filters
    self.alpha = alpha
    self.W = corresponding_unet_filters * alpha
    self.conv2d_bn_1x1 = Conv2d_batchnorm(input_features=input_features,num_of_filters = int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5),
    kernel_size = (1,1),activation='None',padding = 0)

    self.conv2d_bn_3x3 = Conv2d_batchnorm(input_features=input_features,num_of_filters = int(self.W*0.167),
    kernel_size = (3,3),activation='relu',padding = 1)
    self.conv2d_bn_5x5 = Conv2d_batchnorm(input_features=int(self.W*0.167),num_of_filters = int(self.W*0.333),
    kernel_size = (3,3),activation='relu',padding = 1)
    self.conv2d_bn_7x7 = Conv2d_batchnorm(input_features=int(self.W*0.333),num_of_filters = int(self.W*0.5),
    kernel_size = (3,3),activation='relu',padding = 1)
    self.batch_norm1 = nn.BatchNorm2d(int(self.W*0.5)+int(self.W*0.167)+int(self.W*0.333) ,affine=False)

  def forward(self,x: torch.Tensor)->torch.Tensor:

    temp = self.conv2d_bn_1x1(x)
    a = self.conv2d_bn_3x3(x)
    b = self.conv2d_bn_5x5(a)
    c = self.conv2d_bn_7x7(b)
    x = torch.cat([a,b,c],axis=1)
    x = self.batch_norm1(x)
    x = x + temp
    x = self.batch_norm1(x)
    return x

class Conv2d_batchnorm(nn.Module):
  def __init__(self,input_features : int,num_of_filters : int ,kernel_size : Tuple = (2,2),stride : Tuple = (1,1), activation : str = 'relu',padding  : int= 0)->None:
    """
    Arguments:
      x - input layer
      num_of_filters - no. of filter outputs
      filters - shape of the filters to be used
      stride - stride dimension 
      activation -activation function to be used
    Returns - None
    """
    super().__init__()
    self.activation = activation
    self.conv1 = nn.Conv2d(in_channels=input_features,out_channels=num_of_filters,kernel_size=kernel_size,stride=stride,padding = padding)
    self.batchnorm = nn.BatchNorm2d(num_of_filters,affine=False)
  
  def forward(self,x : torch.Tensor)->torch.Tensor:
    x = self.conv1(x)
    x = self.batchnorm(x)
    if self.activation == 'relu':
      return F.relu(x)
    else:
      return x


class Respath(nn.Module):
  def __init__(self,input_features : int,filters : int,respath_length : int)->None:
    """
    Arguments:
    input_features - input layer filters
    filters - output channels
    respath_length - length of the Respath
    
    Returns - None
    """
    super().__init__()
    self.filters = filters
    self.respath_length = respath_length
    self.conv2d_bn_1x1 = Conv2d_batchnorm(input_features=input_features,num_of_filters = self.filters,
    kernel_size = (1,1),activation='None',padding = 0)
    self.conv2d_bn_3x3 = Conv2d_batchnorm(input_features=input_features,num_of_filters = self.filters,
    kernel_size = (3,3),activation='relu',padding = 1)
    self.conv2d_bn_1x1_common = Conv2d_batchnorm(input_features=self.filters,num_of_filters = self.filters,
    kernel_size = (1,1),activation='None',padding = 0)
    self.conv2d_bn_3x3_common = Conv2d_batchnorm(input_features=self.filters,num_of_filters = self.filters,
    kernel_size = (3,3),activation='relu',padding = 1)
    self.batch_norm1 = nn.BatchNorm2d(filters,affine=False)
    
  def forward(self,x : torch.Tensor)->torch.Tensor:
    shortcut = self.conv2d_bn_1x1(x)
    x = self.conv2d_bn_3x3(x)
    x = x + shortcut    
    x = F.relu(x)
    x = self.batch_norm1(x)
    if self.respath_length>1:
      for i in range(self.respath_length):
        shortcut = self.conv2d_bn_1x1_common(x)
        x = self.conv2d_bn_3x3_common(x)
        x = x + shortcut
        x = F.relu(x)
        x = self.batch_norm1(x)
      return x
    else:
      return x

class MultiResUnet(nn.Module):
  def __init__(self,channels : int,filters : int =32,nclasses : int =1)->None:

    """
    Arguments:
    channels - input image channels
    filters - filters to begin with (Unet)
    nclasses - number of classes
    Returns - None
    """
    super().__init__()
    self.alpha = 1.67
    self.filters = filters
    self.nclasses = nclasses
    self.multiresblock1 = Multiresblock(input_features=channels,corresponding_unet_filters=self.filters)
    self.pool1 =  nn.MaxPool2d(2,stride= 2)
    self.in_filters1 = int(self.filters*self.alpha* 0.5)+int(self.filters*self.alpha*0.167)+int(self.filters*self.alpha*0.333)
    self.respath1 = Respath(input_features=self.in_filters1 ,filters=self.filters,respath_length=4)
    self.multiresblock2 = Multiresblock(input_features= self.in_filters1,corresponding_unet_filters=self.filters*2)
    self.pool2 =  nn.MaxPool2d(2, 2)
    self.in_filters2 = int(self.filters*2*self.alpha* 0.5)+int(self.filters*2*self.alpha*0.167)+int(self.filters*2*self.alpha*0.333)
    self.respath2 = Respath(input_features=self.in_filters2,filters=self.filters*2,respath_length=3)
    self.multiresblock3 = Multiresblock(input_features= self.in_filters2,corresponding_unet_filters=self.filters*4)
    self.pool3 =  nn.MaxPool2d(2, 2)
    self.in_filters3 = int(self.filters*4*self.alpha* 0.5)+int(self.filters*4*self.alpha*0.167)+int(self.filters*4*self.alpha*0.333)
    self.respath3 = Respath(input_features=self.in_filters3,filters=self.filters*4,respath_length=2)
    self.multiresblock4 = Multiresblock(input_features= self.in_filters3,corresponding_unet_filters=self.filters*8)
    self.pool4 =  nn.MaxPool2d(2, 2)
    self.in_filters4 = int(self.filters*8*self.alpha* 0.5)+int(self.filters*8*self.alpha*0.167)+int(self.filters*8*self.alpha*0.333)
    self.respath4 = Respath(input_features=self.in_filters4,filters=self.filters*8,respath_length=1)
    self.multiresblock5 = Multiresblock(input_features= self.in_filters4,corresponding_unet_filters=self.filters*16)
    self.in_filters5 = int(self.filters*16*self.alpha* 0.5)+int(self.filters*16*self.alpha*0.167)+int(self.filters*16*self.alpha*0.333)
     
    #Decoder path
    self.upsample6 = nn.ConvTranspose2d(in_channels=self.in_filters5,out_channels=self.filters*8,kernel_size=(2,2),stride=(2,2),padding = 0)  
    self.concat_filters1 = self.filters*8+self.filters*8
    self.multiresblock6 = Multiresblock(input_features=self.concat_filters1,corresponding_unet_filters=self.filters*8)
    self.in_filters6 = int(self.filters*8*self.alpha* 0.5)+int(self.filters*8*self.alpha*0.167)+int(self.filters*8*self.alpha*0.333)
    self.upsample7 = nn.ConvTranspose2d(in_channels=self.in_filters6,out_channels=self.filters*4,kernel_size=(2,2),stride=(2,2),padding = 0)  
    self.concat_filters2 = self.filters*4+self.filters*4
    self.multiresblock7 = Multiresblock(input_features=self.concat_filters2,corresponding_unet_filters=self.filters*4)
    self.in_filters7 = int(self.filters*4*self.alpha* 0.5)+int(self.filters*4*self.alpha*0.167)+int(self.filters*4*self.alpha*0.333)
    self.upsample8 = nn.ConvTranspose2d(in_channels=self.in_filters7,out_channels=self.filters*2,kernel_size=(2,2),stride=(2,2),padding = 0)  
    self.concat_filters3 = self.filters*2+self.filters*2
    self.multiresblock8 = Multiresblock(input_features=self.concat_filters3,corresponding_unet_filters=self.filters*2)
    self.in_filters8 = int(self.filters*2*self.alpha* 0.5)+int(self.filters*2*self.alpha*0.167)+int(self.filters*2*self.alpha*0.333)
    self.upsample9 = nn.ConvTranspose2d(in_channels=self.in_filters8,out_channels=self.filters,kernel_size=(2,2),stride=(2,2),padding = 0)  
    self.concat_filters4 = self.filters+self.filters
    self.multiresblock9 = Multiresblock(input_features=self.concat_filters4,corresponding_unet_filters=self.filters)
    self.in_filters9 = int(self.filters*self.alpha* 0.5)+int(self.filters*self.alpha*0.167)+int(self.filters*self.alpha*0.333)
    self.conv_final = Conv2d_batchnorm(input_features=self.in_filters9,num_of_filters = self.nclasses,
    kernel_size = (1,1),activation='None')

  def forward(self,x : torch.Tensor)->torch.Tensor:
    x_multires1 = self.multiresblock1(x)
    x_pool1 = self.pool1(x_multires1)
    x_multires1 = self.respath1(x_multires1)
    x_multires2 = self.multiresblock2(x_pool1)
    x_pool2 = self.pool2(x_multires2)
    x_multires2 = self.respath2(x_multires2)
    x_multires3 = self.multiresblock3(x_pool2)
    x_pool3 = self.pool3(x_multires3)
    x_multires3 = self.respath3(x_multires3)
    x_multires4 = self.multiresblock4(x_pool3)
    x_pool4 = self.pool4(x_multires4)
    x_multires4 = self.respath4(x_multires4)
    x_multires5 = self.multiresblock5(x_pool4)
    up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
    x_multires6 = self.multiresblock6(up6)
    up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
    x_multires7 = self.multiresblock7(up7)
    up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
    x_multires8 = self.multiresblock8(up8)
    up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
    x_multires9 = self.multiresblock9(up9)
    if self.nclasses > 1:
      conv_final_layer =  self.conv_final(x_multires9)
    else:
      conv_final_layer =  torch.sigmoid(self.conv_final(x_multires9))
    return conv_final_layer