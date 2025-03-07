import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from scipy.stats import kstest
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from tqdm import tqdm
import joblib
import copy

if '../' not in sys.path:
    sys.path = ['../'] + sys.path

current_dir = os.getcwd()
figure_dir = os.path.join(current_dir, 'figures')
model_dir = os.path.join(current_dir, 'models')
data_dir = os.path.join(current_dir, 'datas')
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

font_size = 22
plt.rcParams['axes.titlesize'] = font_size
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size-2
plt.rcParams['ytick.labelsize'] = font_size-2
plt.rcParams['legend.fontsize'] = font_size-6
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'
