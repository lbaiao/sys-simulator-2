from enum import Enum
from sys_simulator.pathloss import pathloss_bs_users
import numpy as np


class node:
    """
    class representing a generic node
    position: x,y tuple representing the BS position coordinates
    radius: BS coverage radius in meters
    """
    def __init__(self, max_power=-7, memory_size=2, **kargs):
        self.tx_power = -1000
        self.sinr = -1000
        self.pathloss_to_bs = 1000
        self.past_actions = -1000 * np.ones(memory_size)
        self.past_sinrs = -1000 * np.ones(memory_size)
        self.past_bs_losses = 1000 * np.ones(memory_size)
        self.past_d2d_losses = 1000 * np.ones(memory_size)
        self.interference_contrib_pct = 0
        self.past_interference_contrib_pct = 0
        self.max_power = max_power
        self.timestep = 0
        self.pathloss_is_set = False
        self.power_is_set = False
        self.sinr_is_set = False
        self.pathloss_d2d_is_set = False
        self.past_interference_contrib_pct_is_set = False

    def set_position(self, position):
        self.position = position

    def set_distance_to_bs(self, distance):
        self.distance_to_bs = distance

    def set_pathloss_to_bs(self, pathloss):
        if self.pathloss_is_set:
            raise Exception(
                'Trying to set the pathloss to BS more \
                than once in the same timestep.'
            )
        aux = np.roll(self.past_bs_losses, 1)
        aux[0] = self.pathloss_to_bs
        self.past_bs_losses = aux
        self.pathloss_to_bs = pathloss
        self.pathloss_is_set = True

    def set_tx_power(self, tx_power):
        if self.power_is_set:
            raise Exception(
                'Trying to set TX power more than once in the same timestep.'
            )
        aux = np.roll(self.past_actions, 1)
        aux[0] = self.tx_power
        self.past_actions = aux
        if tx_power > self.max_power:
            self.tx_power = self.max_power
        else:
            self.tx_power = tx_power
        self.power_is_set = True

    def set_rb(self, rb):
        self.rb = rb

    def set_sinr(self, sinr):
        if self.sinr_is_set:
            raise Exception(
                'Trying to set sinr more than once in the same timestep.'
            )
        aux = np.roll(self.past_sinrs, 1)
        aux[0] = self.sinr
        self.past_sinrs = aux
        self.sinr = sinr
        self.sinr_is_set = True

    def set_gain(self, gain: float):
        self.gain = gain

    def reset_set_flags(self):
        self.pathloss_is_set = False
        self.power_is_set = False
        self.sinr_is_set = False
        self.pathloss_d2d_is_set = False
        self.past_interference_contrib_pct_is_set = False

    def set_interference_contrib_pct(self, contrib: float):
        if self.past_interference_contrib_pct_is_set:
            raise Exception(
                'Trying to set interference_contrib_pct \
                 more than once in the same timestep.'
            )
        self.past_interference_contrib_pct = self.interference_contrib_pct
        self.interference_contrib_pct = contrib
        self.past_interference_contrib_pct_is_set = True


class base_station(node):
    """class representing the base station

    Attributes
    ----------
    position: Tuple[float, float]
        x,y tuple representing the BS position coordinates

    radius: float
        BS coverage radius in meters
    """
    def __init__(self, position, radius=500):
        super(base_station, self).__init__()
        self.position = position
        self.radius = radius
        self.id: str = 'BS:0'

    def set_radius(self, radius):
        self.radius = radius


class mobile_user(node):
    """
    class representing the mobile_user
    position: x,y tuple representing the device position coordinates
    """
    def __init__(self, id, p_max=-7, memory_size=2):
        super(mobile_user, self).__init__(p_max, memory_size)
        self.id: str = f'MUE:{id}'

    def get_tx_power(
            self, bs: base_station, snr: float,
            noise_power: float, margin: float, p_max: float
    ):
        tx_power = snr * noise_power * \
            pathloss_bs_users(self.distance_to_bs/1000) / (self.gain * bs.gain)
        tx_power *= margin
        if tx_power > p_max:
            tx_power = p_max
        return tx_power

    def get_tx_power_db(
            self, bs: base_station, snr: float,
            noise_power: float, margin: float, p_max: float,
    ):
        tx_power = snr + noise_power + \
            self.pathloss_to_bs - \
            (self.gain + bs.gain)
        tx_power += margin
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
                 max_power=-7,
                 memory_size=2, **kwargs):
        super(d2d_user, self).__init__(max_power, memory_size)
        self.type = d2d_type
        self.id: str = f'DUE.{self.type.value}:{id}'
        self.pathloss_d2d = -1000

    def set_distance_d2d(self, distance):
        self.distance_d2d = distance

    def set_pathloss_d2d(self, pathloss):
        if self.pathloss_d2d_is_set:
            raise Exception(
                'Trying to set pathloss to D2D more than \
                 once in the same timestep.'
            )
        aux = np.roll(self.past_d2d_losses, 1)
        aux[0] = self.pathloss_d2d
        self.past_d2d_losses = aux
        self.pathloss_d2d = pathloss

    def set_id_pair(self, id):
        self.id = id

    def set_link_id(self, link_id):
        self.link_id = link_id

    def set_distance_to_mue(self, distance):
        self.distance_to_mue = distance

    @staticmethod
    def get_due_by_id(d2d_list, due_id):
        due = next(x for x in d2d_list if x.id == due_id)
        return due
