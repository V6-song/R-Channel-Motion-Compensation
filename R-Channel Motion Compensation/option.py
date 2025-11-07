import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='./test', help='directory store pre_paired data,include clear and hazy folders') 
parser.add_argument('--results_dir', type=str, default='./test/results', help='directory store results') 
opt=parser.parse_args()
