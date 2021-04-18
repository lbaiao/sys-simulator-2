from copy import copy
from enum import Enum
from sys_simulator.devices.motion_models import MotionModel
from typing import Dict, List, Tuple
from sys_simulator.pathloss import pathloss_bs_users
import numpy as np


def db_to_power(x):
    return 10**(x/10)


class node:
    """
    class representing a generic node
    position: x,y tuple representing the BS position coordinates
    radius: BS coverage radius in meters
    """

    def __init__(
        self,
        max_power=-7,
        memory_size=2,
        beta=.8,
        limited_power=True,
        motion_model='no_movement',
        **kwargs
    ):
        self.position = (0, 0, 0)
        self.tx_power = -100
        self.input_power = -100
        self.limited_power = limited_power
        self.sinr = -100
        self.snr = -100
        self.pathloss_to_bs = 100
        self.beta = beta
        self.past_actions = -100 * np.ones(memory_size)
        self.past_sinrs = -100 * np.ones(memory_size)
        self.past_snrs = -100 * np.ones(memory_size)
        self.past_bs_losses = 100 * np.ones(memory_size)
        self.past_d2d_losses = 100 * np.ones(memory_size)
        self.interference_contrib_pct = 0
        self.past_interference_contrib_pct = 0
        self.max_power = max_power
        self.timestep = 0
        self.pathloss_is_set = False
        self.power_is_set = False
        self.sinr_is_set = False
        self.snr_is_set = False
        self.pathloss_d2d_is_set = False
        self.past_interference_contrib_pct_is_set = False
        self.speffs = 1e-4 * np.ones(memory_size)
        self.max_speffs = 1e-4 * np.ones(memory_size)
        self.speff_is_set = False
        self.max_speff_is_set = False
        self.avg_speffs = 1e-4 * np.ones(memory_size)
        self.avg_speffs_is_set = False
        self.link_prices = 1e-4 * np.ones(memory_size)
        self.link_price_is_set = False
        self.interferences = []
        self.interference_is_set = False
        self.power_at_receiver = -100
        self.power_at_receiver_is_set = False
        self.pathlosses_to_interfering = {}
        self.pathlosses_to_interfering_is_set = False
        self.motion_model = MotionModel(motion_model)
        self.speed = self.motion_model.speed
        self.direction = self.motion_model.direction
        self.has_moved = False

    def set_pathlosses_to_interfering(self, pathlosses: Dict):
        if self.pathlosses_to_interfering_is_set:
            raise Exception(
                'Trying to set pathlosses to interfering D2Ds more than \
                    once in the same timestep.'
            )
        self.pathlosses_to_interfering = pathlosses
        self.pathlosses_to_interfering_is_set = True

    def set_power_at_receiver(self, power: float):
        if self.power_at_receiver_is_set:
            raise Exception(
                'Trying to set power at receiver to D2D more than \
                 once in the same timestep.'
            )
        self.power_at_receiver = power
        self.power_at_receiver_is_set = True

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
        self.pathloss_to_bs = pathloss
        aux = np.roll(self.past_bs_losses, 1)
        aux[0] = self.pathloss_to_bs
        self.past_bs_losses = aux
        self.pathloss_is_set = True

    def set_tx_power(self, tx_power):
        if self.power_is_set:
            raise Exception(
                'Trying to set TX power more than once in the same timestep.'
            )
        self.input_power = copy(tx_power)
        if self.limited_power:
            self.tx_power = \
                tx_power if tx_power < self.max_power else self.max_power
        else:
            self.tx_power = tx_power
        aux = np.roll(self.past_actions, 1)
        aux[0] = self.tx_power
        self.past_actions = aux
        self.power_is_set = True

    def set_rb(self, rb):
        self.rb = rb

    def set_sinr(self, sinr):
        if self.sinr_is_set:
            raise Exception(
                'Trying to set sinr more than once in the same timestep.'
            )
        self.sinr = sinr
        aux = np.roll(self.past_sinrs, 1)
        aux[0] = self.sinr
        self.past_sinrs = aux
        self.sinr_is_set = True

    def set_snr(self, snr: float):
        if self.snr_is_set:
            raise Exception(
                'Trying to set snr more than once in the same timestep.'
            )
        self.snr = snr
        aux = np.roll(self.past_snrs, 1)
        aux[0] = self.snr
        self.past_snrs = aux
        self.snr_is_set = True

    def set_gain(self, gain: float):
        self.gain = gain

    def reset_set_flags(self):
        self.pathloss_is_set = False
        self.power_is_set = False
        self.sinr_is_set = False
        self.snr_is_set = False
        self.pathloss_d2d_is_set = False
        self.past_interference_contrib_pct_is_set = False
        self.avg_speffs_is_set = False
        self.link_price_is_set = False
        self.speff_is_set = False
        self.max_speff_is_set = False
        self.interference_is_set = False
        self.power_at_receiver_is_set = False
        self.pathlosses_to_interfering_is_set = False
        self.has_moved = False

    def set_interference_contrib_pct(self, contrib: float):
        if self.past_interference_contrib_pct_is_set:
            raise Exception(
                'Trying to set interference_contrib_pct \
                 more than once in the same timestep.'
            )
        self.past_interference_contrib_pct = self.interference_contrib_pct
        self.interference_contrib_pct = contrib
        self.past_interference_contrib_pct_is_set = True

    def calculate_avg_speff(self):
        speff = np.log2(1 + db_to_power(self.sinr))
        avg_speff = \
            self.beta * speff + (1 - self.beta) * self.avg_speffs[0]
        return avg_speff

    def set_avg_speff(self, avg_speff: float):
        if self.avg_speffs_is_set:
            raise Exception(
                'Trying to set the average spectral efficiency more \
                than once in the same timestep.'
            )
        aux = np.roll(self.avg_speffs, 1)
        aux[0] = avg_speff
        self.avg_speffs = aux
        self.avg_speffs_is_set = True

    def calc_set_avg_speff_set_link_price(self):
        avg_speff = self.calculate_avg_speff()
        self.set_avg_speff(avg_speff)
        self.set_link_price(1 / avg_speff)

    def set_link_price(self, link_price: float):
        if self.link_price_is_set:
            raise Exception(
                'Trying to set the link price more \
                than once in the same timestep.'
            )
        aux = np.roll(self.link_prices, 1)
        aux[0] = link_price
        self.link_prices = aux
        self.link_price_is_set = True

    def calculate_speff(self) -> float:
        speff = np.log2(1 + db_to_power(self.sinr))
        return speff

    def calculate_max_speff(self) -> float:
        m_speff = np.log2(1 + db_to_power(self.snr))
        return m_speff

    def set_speff(self, speff: float):
        if self.speff_is_set:
            raise Exception(
                'Trying to set the spectral efficiency more \
                than once in the same timestep.'
            )
        aux = np.roll(self.speffs, 1)
        aux[0] = speff
        self.speffs = aux
        self.speff_is_set = True

    def set_max_speff(self, speff: float):
        if self.max_speff_is_set:
            raise Exception(
                'Trying to set the spectral efficiency more \
                than once in the same timestep.'
            )
        aux = np.roll(self.max_speffs, 1)
        aux[0] = speff
        self.max_speffs = aux
        self.max_speff_is_set = True

    def calc_set_speff(self):
        speff = self.calculate_speff()
        self.set_speff(speff)

    def calc_set_max_speff(self):
        m_speff = self.calculate_max_speff()
        self.set_max_speff(m_speff)

    def set_interferences(self, interferences: List[Tuple]):
        if self.interference_is_set:
            raise Exception(
                'Trying to set interferences more \
                than once in the same timestep.'
            )
        self.interferences = interferences
        self.interference_is_set = True

    def calc_received_interference(self):
        _, interferences = [i for i in zip(*self.interferences)]
        total_interference = np.sum(interferences)
        total_interference_db = 10 * np.log10(total_interference)
        return total_interference_db

    def get_received_interference(self, interferer_id: str):
        index = [
            i for i in range(len(self.interferences))
            if self.interferences[i][0] == interferer_id
        ][0]
        interference = self.interferences[index][1]
        return interference

    def norm_speffs(self):
        norm = np.array(self.speffs) / np.array(self.max_speffs)
        return norm

    def move(self, dt: float):
        if self.has_moved:
            raise Exception('Device has already moved.')
        pos, dirc = self.motion_model.step(self.position, self.direction, dt)
        self.position = pos
        self.direction = dirc
        self.has_moved = True


class base_station(node):
    """class representing the base station

    Attributes
    ----------
    position: Tuple[float, float]
        x,y tuple representing the BS position coordinates

    radius: float
        BS coverage radius in meters
    """

    def __init__(self, position, radius=500, motion_model='no_movement'):
        super(base_station, self).__init__(motion_model=motion_model)
        self.set_position(position)
        self.radius = radius
        self.id: str = 'BS:0'

    def set_position(self, position):
        self.position = position

    def set_radius(self, radius):
        self.radius = radius


class mobile_user(node):
    """
    class representing the mobile_user
    position: x,y tuple representing the device position coordinates
    """

    def __init__(self, id, p_max=-7, memory_size=2, motion_model='no_movement'):
        super(mobile_user, self).__init__(p_max, memory_size, motion_model=motion_model)
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
        self, bs: base_station, snr: float, channel_loss: float,
        noise_power: float, margin: float, p_max: float,
    ):
        tx_power = snr + noise_power + \
            channel_loss - \
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
                 memory_size=2, limited_power=True, 
                 motion_model='no_movement', **kwargs):
        super(d2d_user, self).__init__(
            max_power, memory_size, limited_power=limited_power,
            motion_model=motion_model
        )
        self.type = d2d_type
        self.id: str = f'DUE.{self.type.value}:{id}'
        self.pathloss_d2d = -1000
        self.is_dummy = False
        self.received_mue_interference = -100
        self.received_mue_interference_is_set = False
        self.caused_mue_interference = -100
        self.caused_mue_interference_is_set = False

    def set_distance_d2d(self, distance):
        self.distance_d2d = distance

    def set_pathloss_d2d(self, pathloss):
        if self.pathloss_d2d_is_set:
            raise Exception(
                'Trying to set pathloss to D2D more than \
                 once in the same timestep.'
            )
        self.pathloss_d2d = pathloss
        aux = np.roll(self.past_d2d_losses, 1)
        aux[0] = self.pathloss_d2d
        self.past_d2d_losses = aux

    def set_id_pair(self, id):
        self.id = id

    def set_link_id(self, link_id):
        self.link_id = link_id

    def set_distance_to_mue(self, distance):
        self.distance_to_mue = distance

    def set_received_mue_interference(self, interference: float):
        if self.received_mue_interference_is_set:
            raise Exception(
                'Trying to set received mue interference to D2D more than \
                 once in the same timestep.'
            )
        self.received_mue_interference = interference
        self.received_mue_interference_is_set = True

    def set_caused_mue_interference(self, interference: float):
        if self.caused_mue_interference_is_set:
            raise Exception(
                'Trying to set caused mue interference to D2D more than \
                 once in the same timestep.'
            )
        self.caused_mue_interference = interference
        self.caused_mue_interference_is_set = True

    def reset_set_flags(self):
        super(d2d_user, self).reset_set_flags()
        self.caused_mue_interference_is_set = False
        self.received_mue_interference_is_set = False

    @staticmethod
    def get_due_by_id(d2d_list, due_id):
        due = next(x for x in d2d_list if x.id == due_id)
        return due
