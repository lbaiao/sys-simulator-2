from sys_simulator.general.general import db_to_power, power_to_db
from typing import List
from sys_simulator import pathloss
from sys_simulator.devices.devices \
    import d2d_user, mobile_user, d2d_node_type, base_station
from scipy.spatial.distance import euclidean


def sinr_d2d(d2d_tx: d2d_user, d2d_rx: d2d_user, d2d_devices: List[d2d_user],
             mue: mobile_user, noise_power: float, user_gain: float):
    d2d_tx_contrib = d2d_tx.tx_power / \
        pathloss.pathloss_users_db(d2d_tx.distance_d2d/1000) * user_gain**2
    d2d_rx_mue_distance = euclidean(d2d_rx.position, mue.position)
    mue_interference = mue.tx_power / \
        pathloss.pathloss_users(d2d_rx_mue_distance/1000) * user_gain**2
    d2d_interferers = [d for d in d2d_devices if (
        d.id != d2d_tx.id
        and d.type == d2d_node_type.TX and d.rb == d2d_tx.rb)]
    d2d_interference = sum(
        [d.tx_power * user_gain**2 / pathloss.pathloss_users(
            euclidean(d2d_rx.position, d.position)/1000)
            for d in d2d_interferers]
    )
    sinr = d2d_tx_contrib / (noise_power + mue_interference + d2d_interference)
    return sinr


def sinr_d2d_db(
    d2d_tx: d2d_user, d2d_rx: d2d_user, d2d_devices: List[d2d_user],
    mue: mobile_user, noise_power: float, user_gain: float
):
    d2d_tx_contrib = d2d_tx.tx_power - \
        pathloss.pathloss_users_db(d2d_tx.distance_d2d/1000) + 2 * user_gain
    d2d_rx_mue_distance = euclidean(d2d_rx.position, mue.position)
    mue_interference = mue.tx_power - \
        pathloss.pathloss_users_db(d2d_rx_mue_distance/1000) + 2 * user_gain
    d2d_interferers = [d for d in d2d_devices if (
        d.id != d2d_tx.id
        and d.type == d2d_node_type.TX and d.rb == d2d_tx.rb)]
    d2d_interference = sum(
        [
            db_to_power(d.tx_power) *
            db_to_power(2 * user_gain) /
            pathloss.pathloss_users_db(
                euclidean(d2d_rx.position, d.position)/1000
            )
            for d in d2d_interferers
        ]
    )
    sinr = d2d_tx_contrib - \
        power_to_db(
            db_to_power(noise_power) +
            db_to_power(mue_interference) +
            d2d_interference
        )
    return sinr


def sinr_mue(mue: mobile_user, d2d_devices: List[d2d_user], bs: base_station,
             noise_power: float, bs_gain: float, user_gain: float):
    mue_contrib = mue.tx_power * user_gain * bs_gain / \
        pathloss.pathloss_bs_users(mue.distance_to_bs/1000)
    d2d_interferers = [d for d in d2d_devices if (
        d.type == d2d_node_type.TX and d.rb == mue.rb)]
    d2d_interference = sum(
        [d.tx_power * user_gain * bs_gain / pathloss.pathloss_bs_users(
            euclidean(d.position, bs.position)/1000) for d in d2d_interferers]
    )
    sinr = mue_contrib / (noise_power + d2d_interference)
    return sinr


def sinr_mue_db(mue: mobile_user, d2d_devices: List[d2d_user],
                bs: base_station,
                noise_power: float, bs_gain: float, user_gain: float):
    mue_contrib = mue.tx_power + user_gain + bs_gain \
        - pathloss.pathloss_bs_users_db(mue.distance_to_bs/1000)
    d2d_interferers = [d for d in d2d_devices if (
        d.type == d2d_node_type.TX and d.rb == mue.rb)]
    d2d_interference = sum(
        [
            db_to_power(d.tx_power) *
            db_to_power(user_gain) *
            db_to_power(bs_gain) /
            pathloss.pathloss_bs_users(euclidean(d.position, bs.position)/1000)
            for d in d2d_interferers
        ]
    )
    sinr = \
        mue_contrib - power_to_db(db_to_power(noise_power) + d2d_interference)
    return sinr


def sinr_d2d_tensor(d2d_tx: d2d_user, d2d_rx: d2d_user,
                    d2d_devices: List[d2d_user], mue: mobile_user,
                    noise_power: float, user_gain: float):
    d2d_tx_contrib = d2d_tx.tx_power / \
        pathloss.pathloss_users(d2d_tx.distance_d2d/1000) * user_gain**2
    d2d_rx_mue_distance = euclidean(d2d_rx.position, mue.position)
    mue_interference = mue.tx_power / \
        pathloss.pathloss_users(d2d_rx_mue_distance/1000) * user_gain**2
    d2d_interferers = [d for d in d2d_devices if (
        d.id != d2d_tx.id
        and d.type == d2d_node_type.TX and d.rb == d2d_tx.rb)]
    d2d_interference = sum(
        [d.tx_power * user_gain**2 / pathloss.pathloss_users(
            euclidean(d2d_rx.position, d.position)/1000)
            for d in d2d_interferers]
    )
    sinr = d2d_tx_contrib / (noise_power + mue_interference + d2d_interference)
    return sinr


def sinr_mue_tensor(mue: mobile_user, d2d_devices: List[d2d_user],
                    bs: base_station, noise_power: float, bs_gain: float,
                    user_gain: float):
    mue_contrib = mue.tx_power * user_gain * bs_gain / \
        pathloss.pathloss_bs_users(mue.distance_to_bs/1000)
    d2d_interferers = [d for d in d2d_devices if (
        d.type == d2d_node_type.TX and d.rb == mue.rb)]
    d2d_interference = sum(
        [d.tx_power * user_gain * bs_gain / pathloss.pathloss_bs_users(
            euclidean(d.position, bs.position)/1000)
            for d in d2d_interferers]
    )
    sinr = mue_contrib / (noise_power + d2d_interference)
    return sinr
