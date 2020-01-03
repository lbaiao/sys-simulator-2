import sys
sys.path.insert(1, 'D:\\Dev\\sys-simulator-2')

from typing import List
from pathloss import pathloss
from devices.devices import d2d_user, mobile_user, d2d_node_type, base_station

from scipy.spatial.distance import euclidean


def sinr_d2d(d2d_tx: d2d_user, d2d_devices: List[d2d_user], mue: mobile_user, noise_power: float):
    d2d_tx_contrib = d2d_tx.tx_power * pathloss.pathloss_users(d2d_tx.distance_d2d)
    d2d_mue_distance = euclidean(d2d_tx.position, mue.position)
    mue_interference = mue.tx_power * pathloss.pathloss_users(d2d_mue_distance)
    d2d_interferers = [d for d in d2d_devices if (d.id != d2d_tx.id and d.type == d2d_node_type.TX and d.rb == d2d_tx.rb)]
    d2d_interference = sum([d.tx_power * pathloss.pathloss_users(euclidean(d2d_tx.position, d.position)) for d in d2d_interferers])
    sinr = d2d_tx_contrib / (noise_power + mue_interference + d2d_interference)
    return sinr


def sinr_mue(mue: mobile_user, d2d_devices: List[d2d_user], bs: base_station, noise_power: float):
    mue_contrib = mue.tx_power * pathloss.pathloss_bs_users(mue.distance_to_bs)
    d2d_interferers = [d for d in d2d_devices if (d.type == d2d_node_type.TX and d.rb == mue.rb)]
    d2d_interference = sum([d.tx_power * pathloss.pathloss_bs_users(euclidean(d.position, bs.position)) for d in d2d_interferers])
    sinr = mue_contrib / (noise_power + d2d_interference)
    return sinr