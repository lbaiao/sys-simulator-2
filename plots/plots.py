import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)
# sys.path.insert(1, 'D:\\Dev\\sys-simulator-2')

from typing import List
import matplotlib.pyplot as plt
from devices.devices import base_station, mobile_user, d2d_user

def plot_positions(bs: base_station, mues: List[mobile_user], d2d_txs: List[d2d_user], d2d_rxs: List[d2d_user]):
    bs_x = bs.position[0]
    bs_y = bs.position[1]
    mues_x = [i.position[0] for i in mues]
    mues_y = [i.position[1] for i in mues]
    d2d_txs_x = [i.position[0] for i in d2d_txs]
    d2d_txs_y = [i.position[1] for i in d2d_txs]
    d2d_rxs_x = [i.position[0] for i in d2d_rxs]
    d2d_rxs_y = [i.position[1] for i in d2d_rxs]

    plt.figure()

    p0 = plt.plot(bs_x, bs_y, '*', label='BS')
    p1 = plt.plot(mues_x, mues_y, '*', label='MUEs')
    p2 = plt.plot(d2d_txs_x, d2d_txs_y, 'd', label='D2D TX')
    p3 = plt.plot(d2d_rxs_x, d2d_rxs_y, 'd', label='D2D RX')    

    patch = plt.Circle(bs.position, bs.radius, edgecolor='red', facecolor='None', linewidth=1.0, zorder=10)
    ax = plt.gca()
    ax.add_patch(patch)
    plt.xlim(left=-bs.radius-50)
    plt.xlim(right=bs.radius+50)
    plt.ylim(bottom=-bs.radius-50)
    plt.ylim(top=bs.radius+50)
    plt.title('Nodes')
    plt.legend()
    plt.show()