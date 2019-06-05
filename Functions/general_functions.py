# General arr lib
import numpy as np

# Compile py code
from numba import jit
from numba import prange

# Multithreading
import dask.array as da
import dask as dk
dk.config.set(scheduler='processes')

from dask_image.ndfilters import generic_filter as d_gf

from collections import deque

from skimage.filters import gabor
from skimage.restoration import  denoise_bilateral

import datetime

import numpy as np
from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage.filters import generic_filter as gf
import scipy.stats.mstats as ms
from scipy.stats import skew
import scipy.ndimage.morphology as morph
from scipy import ndimage
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import random
import math
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.filters import gabor
from skimage.util import random_noise
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, confusion_matrix, precision_score
from numba import jit
from numba import prange



@jit
def create_circular_mask(radius):
    """
    Creates a circular mask.
    """
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    mask[radius][radius] = 0
    kernel[mask] = 1
    return kernel

def extract_numpy_files_in_folder(path, skip=[]):
    """
    Returns all the .npy files in a given directory.
    Skips subdirectories.
    """
    root, _, files = next(os.walk(path))
    holder = []
        
    for file in files:
        if file[-3:] != ".npy":
            continue
        elif file in skip:
            continue
        holder.append(os.path.join(root,file))
    return holder

def generate_mask(small_radius, big_radius):
    """
    Generate a mask in a circular radius around a point.
    """
    height, width = big_radius*2,big_radius*2
    Y, X = np.ogrid[:height+1, :width+1]
    distance_from_center = np.sqrt((X- big_radius)**2 + (Y-big_radius)**2)
    mask = (small_radius <= distance_from_center) & (distance_from_center <= big_radius)
    return mask

def circle_mask_generator(radius):
    """
    Wrapper around generate_mask for usage when only a circular radius.
    """
    return generate_mask(1,radius)

def create_filter_with_mask(postfix, arr_with_filenames, function, mask):
    """
    Create a filter over an array of filenames.npy files.
    Existing files with correct naming schemes will NOT be updated if they exist.
    _raw files will be skipped.
    Returns an iterator that can be used to show/save filtered arrays. A name is also yielded.
    """
    for filename in arr_with_filenames:
        if filename[-4:] != "_raw":
            continue
        elif os.path.isfile(f"./{filename[:-4]}_{postfix}.npy"):
            continue
        arr = np.load(f"{filename}.npy")
        holder = gf(arr, function, footprint=mask)
        yield (f"{filename[:-4]}_{postfix}", holder)

def merge_numpy_zones_files(list_of_files):
    """
    Takes a list of paths to files and loads them into a panda DataFrame.
    If the name contains the word 'ditches', the name is replaced by 'labels'.
    Only one zone should be contained inside the list.
    """
    holder = {}
    for file in list_of_files:
        if "ditches" in file or "Ditches" in file:
            holder["labels"] = np.load(file).reshape(-1)
        else:
            holder [file.split("/")[-1][:-4]] = np.load(file).reshape(-1)
    return pd.DataFrame(data=holder)

def create_balanced_mask(ditchArr, height, width):
    """
    Creates a mask from a labeled zone to balance the ditch and non-ditch classes more.
    """
    new_arr = ditchArr.copy()
    for i in range(0, len(ditchArr), height):
        for j in range(0, len(ditchArr[i]), width):
            zoneContainsDitches = None
            if (random.random() * 100 > 99):
                zoneContainsDitches = True
            for k in range(height):
                for l in range(width):
                    if ditchArr[i+k][j+l] == 1:
                        zoneContainsDitches = True
                    if zoneContainsDitches == True:
                        for m in range(height):
                            for n in range(width):
                                new_arr[i+m][j+n] = 1
                    if zoneContainsDitches == True:
                        break
                if zoneContainsDitches == True:
                    break
            if zoneContainsDitches == None:
                for m in range(height):
                    for n in range(width):
                        new_arr[i+m][j+n] = 0
    return new_arr

def create_balanced_dataset(list_of_zones):
    """
    Takes a list of zone file paths and generates a dataset based on balanced masks from the labels for the different zones.
    """
    frames = np.empty(len(list_of_zones), dtype=object)
    for i, zone_file_name in enumerate(list_of_zones):
        zone = pd.read_pickle(zone_file_name)
        mask = masks[i] 
        
        mask = create_balanced_mask(zone["label_3m"].values.reshape((2997,2620)), 3, 5)
        
        mask = np.where(np.invert(mask.reshape(-1)))
        
        zone.drop(zone.index[mask], inplace=True)
        frames[i] = zone
    return pd.concat(frames)

def yield_training_test_zones(list_of_files):
    folds = len(list_of_files)
    arr = list_of_files.copy()
    for i in range(folds):
      training, testing = np.delete(arr, i, 0), arr[i]
      yield (training, testing)