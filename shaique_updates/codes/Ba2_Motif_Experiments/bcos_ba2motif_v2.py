import os
import tqdm
import random
import math
import sys
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.loader import DataLoader
from torch.nn import Module, ModuleList, Sequential, Dropout
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root for bcos module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bcos.modules import BcosLinear
from bcos.modules.norms import DetachableLayerNorm
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.aggr import MeanAggregation