

import cv2, os
import pandas as pd
import numpy as np
import random
import pandas as pd
import  numpy as np
import os
import torch.utils.data as torchdata
from . import shanghai


import torch 
import torchvision.transforms.functional as FT
import copy

from torchvision import transforms
import collections
import torch
import numpy as np
import random
from PIL import ImageOps


def get_dataset(dataset_dict, split, datadir, exp_dict, dataset_size=None):
    name = dataset_dict['name']
    if name == 'shanghai':
        dataset = shanghai.Shanghai(split, datadir=datadir, exp_dict=exp_dict)
    else:
        raise ValueError('dataset %s not defined.' % name)

    return dataset