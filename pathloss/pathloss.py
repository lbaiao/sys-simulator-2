import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

import numpy as np

def pathloss_bs_users(d):
    """
    d: distance between user and bs in km
    returns the pathloss
    """
    pathloss = 15.3+37.6*np.log10(d) #  pathloss in db
    pathloss = 10**(pathloss/10)
    return pathloss

def pathloss_users(d):
    """
    d: distance between users in km
    returns the pathloss
    """
    pathloss = 28+40*np.log10(d) #  pathloss in db
    pathloss = 10**(pathloss/10)
    return pathloss