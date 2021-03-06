import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from typing import List
from pathloss import pathloss
from devices.devices import d2d_user, mobile_user, d2d_node_type, base_station

from scipy.spatial.distance import euclidean

# TODO: a potencia do ruido deve ser multiplicada pelo ganho da antena receptora?
# TODO: o ganho deve ser levado em conta também na transmissão?
def sinr_d2d(d2d_tx: d2d_user, d2d_rx: d2d_user, d2d_devices: List[d2d_user], mue: mobile_user, noise_power: float, user_gain: float):    
    d2d_tx_contrib = d2d_tx.tx_power / pathloss.pathloss_users(d2d_tx.distance_d2d/1000) * user_gain**2
    d2d_rx_mue_distance = euclidean(d2d_rx.position, mue.position)
    mue_interference = mue.tx_power / pathloss.pathloss_users(d2d_rx_mue_distance/1000) * user_gain**2
    d2d_interferers = [d for d in d2d_devices if (d.id != d2d_tx.id and d.type == d2d_node_type.TX and d.rb == d2d_tx.rb)]
    d2d_interference = sum([d.tx_power * user_gain**2 / pathloss.pathloss_users(euclidean(d2d_rx.position, d.position)/1000) for d in d2d_interferers])
    sinr = d2d_tx_contrib / (noise_power + mue_interference + d2d_interference)
    return sinr


def sinr_mue(mue: mobile_user, d2d_devices: List[d2d_user], bs: base_station, noise_power: float, bs_gain: float, user_gain: float):
    mue_contrib = mue.tx_power * user_gain * bs_gain / pathloss.pathloss_bs_users(mue.distance_to_bs/1000)
    d2d_interferers = [d for d in d2d_devices if (d.type == d2d_node_type.TX and d.rb == mue.rb)]
    d2d_interference = sum([d.tx_power * user_gain * bs_gain / pathloss.pathloss_bs_users(euclidean(d.position, bs.position)/1000) for d in d2d_interferers])
    sinr = mue_contrib / (noise_power + d2d_interference)
    return sinr


def sinr_d2d_tensor(d2d_tx: d2d_user, d2d_rx: d2d_user, d2d_devices: List[d2d_user], mue: mobile_user, noise_power: float, user_gain: float):    
    d2d_tx_contrib = d2d_tx.tx_power / pathloss.pathloss_users(d2d_tx.distance_d2d/1000) * user_gain**2
    d2d_rx_mue_distance = euclidean(d2d_rx.position, mue.position)
    mue_interference = mue.tx_power / pathloss.pathloss_users(d2d_rx_mue_distance/1000) * user_gain**2
    d2d_interferers = [d for d in d2d_devices if (d.id != d2d_tx.id and d.type == d2d_node_type.TX and d.rb == d2d_tx.rb)]
    d2d_interference = sum([d.tx_power * user_gain**2 / pathloss.pathloss_users(euclidean(d2d_rx.position, d.position)/1000) for d in d2d_interferers])
    sinr = d2d_tx_contrib / (noise_power + mue_interference + d2d_interference)
    return sinr


def sinr_mue_tensor(mue: mobile_user, d2d_devices: List[d2d_user], bs: base_station, noise_power: float, bs_gain: float, user_gain: float):
    mue_contrib = mue.tx_power * user_gain * bs_gain / pathloss.pathloss_bs_users(mue.distance_to_bs/1000)
    d2d_interferers = [d for d in d2d_devices if (d.type == d2d_node_type.TX and d.rb == mue.rb)]
    d2d_interference = sum([d.tx_power * user_gain * bs_gain / pathloss.pathloss_bs_users(euclidean(d.position, bs.position)/1000) for d in d2d_interferers])
    sinr = mue_contrib / (noise_power + d2d_interference)
    return sinr