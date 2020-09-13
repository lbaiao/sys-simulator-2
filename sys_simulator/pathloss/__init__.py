import numpy as np


def pathloss_bs_users(d):
    """
    d: distance between user and bs in km
    returns the pathloss, in magnitude
    """
    pathloss = 15.3+37.6*np.log10(d)  # pathloss in db
    pathloss = 10**(pathloss/10)
    return pathloss


def pathloss_bs_users_db(d):
    """
    d: distance between user and bs in km
    returns the pathloss, in dB
    """
    pathloss = 15.3+37.6*np.log10(d)  # pathloss in db
    return pathloss


def pathloss_users(d):
    """
    d: distance between users in km
    returns the pathloss, in magnitude
    """
    pathloss = 28+40*np.log10(d)  # pathloss in db
    pathloss = 10**(pathloss/10)
    return pathloss


def pathloss_users_db(d):
    """
    d: distance between users in km
    returns the pathloss, in dB
    """
    pathloss = 28+40*np.log10(d)  # pathloss in db
    return pathloss
