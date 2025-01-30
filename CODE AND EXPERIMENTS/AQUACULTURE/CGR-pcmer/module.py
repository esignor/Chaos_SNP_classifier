import collections
import csv
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib import cm
import pylab
import math
import numpy as np
from PIL import Image
import os
from os import listdir
from PIL import Image
from numpy import asarray
from sklearn.model_selection import StratifiedKFold
from keras.layers import Convolution1D, MaxPooling1D, Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.initializers import glorot_uniform
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import classification_report, confusion_matrix
from os.path import isfile, join
import time
import keras.layers as layers
import glob
import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import multiprocessing as mp
import tensorflow as tf
from itertools import product
from sklearn.model_selection import train_test_split
import threading
import multiprocessing
from keras import regularizers
import shap # for Shapley values


