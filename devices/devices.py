import sys
import os
# lucas_path = os.environ['LUCAS_PATH']
# sys.path.insert(1, lucas_path)

from enum import Enum
from pathloss.pathloss import pathloss_bs_users
import numpy as np
import scipy.spatial as spatial

class node:
    """
    class representing a generic node
    position: x,y tuple representing the BS position coordinates
    radius: BS coverage radius in meters
    """
    def __init__(self, **kargs):
        self.tx_power = 1e-9
        pass

    def set_position(self, position):
        self.position = position

    def set_distance_to_bs(self, distance):
        self.distance_to_bs = distance

    def set_pathloss_to_bs(self, pathloss):
        self.pathloss_to_bs = pathloss

    def set_tx_power(self, tx_power):
        self.tx_power = tx_power

    def set_rb(self, rb):
        self.rb = rb

    def set_sinr(self, sinr):
        self.sinr = sinr

    def set_gain(self, gain: float):
        self.gain = gain


class base_station(node):
    """
    class representing the base station
    position: x,y tuple representing the BS position coordinates
    radius: BS coverage radius in meters
    """
    def __init__(self, position, radius = 500):
        super(base_station, self).__init__()
        self.position = position
        self.radius = radius
    def set_radius(self, radius):
        self.radius = radius

class mobile_user(node):
    """
    class representing the mobile_user
    position: x,y tuple representing the device position coordinates    
    """
    def __init__(self, id):
        super(mobile_user, self).__init__()
        self.id = f'MUE:{id}'

    def get_tx_power(self, bs: base_station, snr: float, noise_power: float, margin: float, p_max: float):
        tx_power = snr * noise_power * pathloss_bs_users(self.distance_to_bs)/ (self.gain * bs.gain)
        tx_power *= margin
        if tx_power > p_max:
            tx_power = p_max
        return tx_power


class d2d_node_type(Enum):
    TX = 'TX'
    RX = 'RX'


class d2d_user(node):
    """
    class representing the d2d_user
    position: x,y tuple representing the
    device position coordinates
    """
    def __init__(self, id: int, d2d_type: d2d_node_type,
                 max_power=0.19952623149688797, **kwargs):
        super(d2d_user, self).__init__()
        self.type = d2d_type
        self.id = f'DUE.{self.type.value}:{id}',
        self.max_power = max_power

    def set_distance_d2d(self, distance):
        self.distance_d2d = distance

    def set_pathloss_d2d(self, pathloss):
        self.pathloss_d2d = pathloss

    def set_id_pair(self, id):
        self.id = id

    def set_link_id(self, link_id):
        self.link_id = link_id

    def set_distance_to_mue(self, distance):
        self.distance_to_mue = distance

    def set_tx_power(self, power):
        if power < 0:
            self.tx_power = 0
        elif power > self.max_power:
            self.tx_power = self.max_power
        else:
            self.tx_power = power

    @staticmethod
    def get_due_by_id(d2d_list, due_id):
        due = next(x for x in d2d_list if x.id == due_id)
        return due
