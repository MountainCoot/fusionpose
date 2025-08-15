import shutil
import os
import rosbag
import subprocess           
import math
import numpy as np


def keep_max_n_files(file_name: str, n=4) -> None:
    # save to file_name if does not exist, otherwise rename all files by increasing their counter
    # should be sturcutere as follows: file_name, file_name_1, file_name_2, file_name_3, ...
    ext = os.path.splitext(file_name)[1]
    if not os.path.exists(file_name):
        return 
    # iterate from back by deleting limit file, and renaming all others
    file_name_strp = os.path.splitext(file_name)[0]
    for i in range(n-1, -1, -1):
        file_name_i = f'{file_name_strp}_{i}{ext}' if i > 0 else file_name
        if i == n-1 and os.path.exists(file_name_i):
            os.remove(file_name_i)
        else:
            file_name_i_1 = f'{file_name_strp}_{i+1}{ext}'
            if os.path.exists(file_name_i):
                shutil.copy(file_name_i, file_name_i_1)



def load_rosbag(path: str) -> rosbag.Bag:
    # check if rosbag exists, and try to open it
    if not os.path.exists(path):
        # check if with .active extension
        path_active = path + '.active'
        if not os.path.exists(path_active):
            raise ValueError(f'Rosbag file {path} does not exist')
        try:
            bag = rosbag.Bag(path_active)
        except rosbag.bag.ROSBagUnindexedException:
            print(f'Rosbag file {path_active} is unindexed, indexing...')
            # try to reindex
            subprocess.run(['rosbag', 'reindex', path_active])
            # rename to original
            os.rename(path_active, path)
            bag = rosbag.Bag(path)
            # remove orig.active file
            path_active_orig = path_active.replace('.active', '.orig.active')
            if os.path.exists(path_active_orig):
                print(f'Removing {path_active_orig}')
                os.remove(path_active_orig)
    else:
        try:
            bag = rosbag.Bag(path)
        except rosbag.bag.ROSBagUnindexedException:
            print(f'Rosbag file {path} is unindexed, indexing...')
            # try to reindex
            subprocess.run(['rosbag', 'reindex', path])
            bag = rosbag.Bag(path)
            # remove active.orig file
            path_orig = path.replace('.bag', '.orig.bag')
            if os.path.exists(path_orig):
                print(f'Removing {path_orig}')
                os.remove(path_orig)
    return bag


def vector_rms(arr: np.ndarray, axis: int) -> float:
    """Computes the RMS magnitude of an array of vectors."""
    return math.sqrt(np.mean(np.sum(np.square(arr), axis=axis)))

def clamp(x: float, xmin: float, xmax: float) -> float:
    return max(min(x, xmax), xmin)
