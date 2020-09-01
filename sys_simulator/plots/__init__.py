from typing import List
import matplotlib.pyplot as plt
import numpy as np
from sys_simulator.devices.devices import base_station, mobile_user, d2d_user
from sys_simulator.q_learning.environments.environment import RLEnvironment


def plot_positions(bs: base_station, mues: List[mobile_user],
                   d2d_txs: List[d2d_user], d2d_rxs: List[d2d_user]):
    bs_x = bs.position[0]
    bs_y = bs.position[1]
    mues_x = [i.position[0] for i in mues]
    mues_y = [i.position[1] for i in mues]
    d2d_txs_x = [i.position[0] for i in d2d_txs]
    d2d_txs_y = [i.position[1] for i in d2d_txs]
    d2d_rxs_x = [i.position[0] for i in d2d_rxs]
    d2d_rxs_y = [i.position[1] for i in d2d_rxs]

    plt.figure()

    _ = plt.plot(bs_x, bs_y, '*', label='BS')
    _ = plt.plot(mues_x, mues_y, '*', label='MUEs')
    _ = plt.plot(d2d_txs_x, d2d_txs_y, 'd', label='D2D TX')
    _ = plt.plot(d2d_rxs_x, d2d_rxs_y, 'd', label='D2D RX')

    coordinates = [(str(i), d2d_txs_x[i], d2d_txs_y[i])
                   for i in range(len(d2d_txs))]
    for c in coordinates:
        plt.annotate(c[0], (c[1], c[2]))

    patch = plt.Circle(bs.position, bs.radius,
                       edgecolor='red', facecolor='None',
                       linewidth=1.0, zorder=10)
    ax = plt.gca()
    ax.add_patch(patch)
    plt.xlim(left=-bs.radius-50)
    plt.xlim(right=bs.radius+50)
    plt.ylim(bottom=-bs.radius-50)
    plt.ylim(top=bs.radius+50)
    plt.title('Nodes')
    plt.legend()
    plt.show()


def plot_spectral_effs(env: RLEnvironment):
    threshold_eff = \
        np.log2(1 + env.params.sinr_threshold) * \
        np.ones(len(env.d2d_spectral_eff))
    plt.figure()
    x_axis = range(len(env.d2d_spectral_eff))
    plt.plot(x_axis, env.d2d_spectral_eff, '.', label='D2D')
    plt.plot(x_axis, env.mue_spectral_eff, '.', label='MUE')
    plt.plot(x_axis, threshold_eff, label='Threshold')
    plt.title('Total spectral efficiencies')
    plt.legend()
    plt.show()


def plot_positions_and_actions(
    bs: base_station, mues: List[mobile_user],
    d2d_txs: List[d2d_user], d2d_rxs: List[d2d_user],
    actions_indexes: List[int]
):
    bs_x = bs.position[0]
    bs_y = bs.position[1]
    mues_x = [i.position[0] for i in mues]
    mues_y = [i.position[1] for i in mues]
    d2d_txs_x = [i.position[0] for i in d2d_txs]
    d2d_txs_y = [i.position[1] for i in d2d_txs]
    d2d_rxs_x = [i.position[0] for i in d2d_rxs]
    d2d_rxs_y = [i.position[1] for i in d2d_rxs]

    plt.figure()

    _ = plt.plot(bs_x, bs_y, '*', label='BS')
    _ = plt.plot(mues_x, mues_y, '*', label='MUEs')
    _ = plt.plot(d2d_txs_x, d2d_txs_y, 'd', label='D2D TX')
    _ = plt.plot(d2d_rxs_x, d2d_rxs_y, 'd', label='D2D RX')

    coordinates = [
        (actions_indexes[i], d2d_txs_x[i], d2d_txs_y[i])
        for i in range(len(d2d_txs))
    ]
    for c in coordinates:
        plt.annotate(c[0], (c[1], c[2]))

    patch = plt.Circle(
        bs.position, bs.radius, edgecolor='red',
        facecolor='None', linewidth=1.0, zorder=10
    )
    ax = plt.gca()
    ax.add_patch(patch)
    plt.xlim(left=-bs.radius-50)
    plt.xlim(right=bs.radius+50)
    plt.ylim(bottom=-bs.radius-50)
    plt.ylim(top=bs.radius+50)
    plt.title(f'N={len(actions_indexes)}')
    plt.legend()


def pie_plot(values: List[float], title: str):
    labels = [f'D2D {i}' for i in range(len(values))]
    _, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title(title)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')


def plot_positions_actions_pie(
    bs: base_station, mue: mobile_user,
    d2d_txs: List[d2d_user], d2d_rxs: List[d2d_user],
    actions_indexes: List[int], values: List[float],
    mue_success: int, mue_sinr_threshold: float,
    reward: float
):
    bs_x = bs.position[0]
    bs_y = bs.position[1]
    mues_x = mue.position[0]
    mues_y = mue.position[1]
    d2d_txs_x = [i.position[0] for i in d2d_txs]
    d2d_txs_y = [i.position[1] for i in d2d_txs]
    d2d_rxs_x = [i.position[0] for i in d2d_rxs]
    d2d_rxs_y = [i.position[1] for i in d2d_rxs]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    _ = ax1.plot(bs_x, bs_y, '*', label='BS')
    _ = ax1.plot(mues_x, mues_y, '*', label='MUEs')
    _ = ax1.plot(d2d_txs_x, d2d_txs_y, 'd', label='D2D TX')
    _ = ax1.plot(d2d_rxs_x, d2d_rxs_y, 'd', label='D2D RX')

    coordinates = [
        (f'({i},{actions_indexes[i]})', d2d_txs_x[i], d2d_txs_y[i])
        for i in range(len(d2d_txs))
    ]
    for c in coordinates:
        ax1.annotate(c[0], (c[1], c[2]))

    patch = plt.Circle(
        bs.position, bs.radius, edgecolor='red',
        facecolor='None', linewidth=1.0, zorder=10
    )
    _ = plt.gca()
    ax1.add_patch(patch)
    ax1.set_xlim(left=-bs.radius-50)
    ax1.set_xlim(right=bs.radius+50)
    ax1.set_ylim(bottom=-bs.radius-50)
    ax1.set_ylim(top=bs.radius+50)
    ax1.set_title(
        f'N={len(actions_indexes)}, M_S={bool(mue_success)} \
        (pair_id, action_id)'
    )
    ax1.legend()

    labels = [f'D2D {i}' for i in range(len(values))]

    # fig1, ax1 = plt.subplots()
    ax2.pie(values, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.set_title(
        f'MUE SINR: {mue.sinr:.2f} | min SINR: {mue_sinr_threshold:.2f} \
        reward: {reward:.4f}'
    )
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_aspect('equal')
