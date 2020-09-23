import time
import numpy as np
import random
import math
import ntpath
from typing import List
import scipy.spatial as spatial
from torch import device
from sys_simulator.devices.devices import d2d_user, mobile_user, base_station


def bits_gen(n):
    return [random.randint(0, 1) for b in range(1, n+1)]


def db_to_power(x):
    return 10**(x/10)


def power_to_db(x):
    return 10*np.log10(x)


def upsample(input, factor):
    z_mat = np.zeros([factor-1, len(input[0])])
    aux = np.concatenate((input, z_mat), axis=0)
    aux2 = np.transpose(aux)
    output = np.reshape(aux2, (1, len(input[0])*factor))
    return output


def downsample(input, factor):
    output = []
    for i in range(0, len(input)):
        if i % factor == 0:
            output.append(input[i])
    return output


def ber(tx_signal, rx_signal):
    return np.sum(np.abs(tx_signal - rx_signal))/len(tx_signal)


def bpsk_theoric(snr):
    # snr in dB
    snr_mag = [10**(x/10) for x in snr]
    return [0.5*math.erfc(np.sqrt(i)) for i in snr_mag]


def distribute_users(mobile_users: List[mobile_user], d2d_users: List[d2d_user], base_station: base_station):
    center = base_station.position
    radius = base_station.radius
    for m in mobile_users:
        x = (np.random.rand()-0.5)*2*radius+center[0]
        y = (np.random.rand()-0.5)*2*(1-np.sqrt(radius**2-x**2))+center[1]
        m.set_position((x,y))   
    for d in d2d_users:
        x = (np.random.rand()-0.5)*2*radius+center[0]
        y = (np.random.rand()-0.5)*2*(1-np.sqrt(radius**2-x**2))+center[1]
        d.set_position((x,y))   


def distribute_nodes(nodes, base_station):
    center = base_station.position
    radius = base_station.radius
    for n in nodes:
        x = (np.random.rand()-0.5)*2*radius+center[0]
        y = (np.random.rand()-0.5)*2*(1-np.sqrt(radius**2-x**2))+center[1]
        n.set_position((x, y))
        n.set_distance_to_bs(
            spatial.distance.euclidean(n.position, base_station.position)
        )


def distribute_mue_validation(nodes: List[mobile_user], base_station):
    if len(nodes) != 1:
        raise 'number of mues must be 1'
    if base_station.position != (0,0):
        raise 'BS position must be (0,0)'
    center = base_station.position
    nodes[0].set_position((center[0], center[1]+100))
    nodes[0].set_distance_to_bs(spatial.distance.euclidean(nodes[0].position, base_station.position))


def distribute_pair_fixed_distance_multiple(nodes_tx: List[d2d_user], nodes_rx: List[d2d_user], base_station):
    """
    Distribute d2d pairs. Nodes_tx and nodes_rx should be lists with the length
    """    
    for i in range(len(nodes_tx)):
        center = base_station.position
        radius = base_station.radius
        is_node2_in_circle = False  
        x1 = (np.random.rand()-0.5)*2*radius+center[0]
        y1 = (np.random.rand()-0.5)*2*(1-np.sqrt(radius**2-x1**2))+center[1]
        nodes_tx[i].set_position((x1,y1))           
        nodes_tx[i].set_distance_to_bs(spatial.distance.euclidean(center, nodes_tx[i].position))
        while(not is_node2_in_circle):
            angle = np.random.rand()*2*np.pi
            x2 = (np.random.rand()-0.5)*2*nodes_tx[i].distance_d2d+x1
            y2 = nodes_tx[i].distance_d2d*np.sin(angle)+y1
            nodes_bs_distance = spatial.distance.euclidean((x2,y2), base_station.position)        
            if nodes_bs_distance < radius:
                nodes_rx[i].set_position((x2,y2))
                nodes_rx[i].set_distance_to_bs(nodes_bs_distance)
                is_node2_in_circle = True        


def distribute_pair_fixed_distance(nodes, base_station, pair_distance):
    center = base_station.position
    radius = base_station.radius
    is_node2_in_circle = False  
    x1 = (np.random.rand()-0.5)*2*radius+center[0]
    y1 = (np.random.rand()-0.5)*2*(1-np.sqrt(radius**2-x1**2))+center[1]
    nodes[0].set_position((x1,y1))           
    nodes[0].set_distance_to_bs(spatial.distance.euclidean(center, nodes[0].position))
    while(not is_node2_in_circle):
        angle = np.random.rand()*2*np.pi
        x2 = pair_distance*np.cos(angle) + x1
        y2 = pair_distance*np.sin(angle) + y1
        nodes_bs_distance = spatial.distance.euclidean((x2,y2), base_station.position)        
        if nodes_bs_distance < radius:
            nodes[1].set_position((x2,y2))
            nodes[1].set_distance_to_bs(nodes_bs_distance)
            # print(spatial.distance.euclidean(nodes[0].position, nodes[1].position))
            is_node2_in_circle = True 


def distribute_rx_fixed_distance(nodes: device, base_station: base_station,
                                 pair_distance: float):
    radius = base_station.radius
    is_node2_in_circle = False
    x1 = nodes[0].position[0]
    y1 = nodes[0].position[1]
    while(not is_node2_in_circle):
        angle = np.random.rand()*2*np.pi
        x2 = pair_distance*np.cos(angle) + x1
        y2 = pair_distance*np.sin(angle) + y1
        nodes_bs_distance = spatial.distance.euclidean((x2, y2),
                                                       base_station.position)
        if nodes_bs_distance < radius:
            nodes[1].set_position((x2, y2))
            nodes[1].set_distance_to_bs(nodes_bs_distance)
            is_node2_in_circle = True


def distribute_d2d_validation(pairs: List[List[d2d_user]], base_station: base_station):
    if len(pairs) != 4:
        raise 'number of mues must be 4'
    if base_station.position != (0,0):
        raise 'BS position must be (0,0)'

    pairs[0][0].set_position((-250,250))
    pairs[0][1].set_position((-250,300))
    pairs[1][0].set_position((-250,-250))
    pairs[1][1].set_position((-250,-300))
    pairs[2][0].set_position((250,-250))
    pairs[2][1].set_position((250,-300))
    pairs[3][0].set_position((250,250))
    pairs[3][1].set_position((250,300))

    for p in pairs:
        for n in p:
            n.set_distance_to_bs(spatial.distance.euclidean(n.position, base_station.position))
            n.set_distance_d2d(50)            


def get_distances_table(nodes):
    distances_table = [ [spatial.distance.euclidean(node.position, i.position) for i in nodes] for node in nodes]
    return np.array(distances_table)


def get_d2d_links(d2d_nodes_distances_table, d2d_nodes, channel):
    it_index = [i for i in range(d2d_nodes_distances_table.shape[0])]
    smallest_distance = {'table_position': (99,99), 'distance': 1e6}    
    d2d_pairs_table = dict()
    d2d_pairs_pathloss_table = dict()
    d2d_pairs_index = 0
    while(len(it_index)>0):
        for i in it_index:
            for j in it_index:
                if smallest_distance['distance'] >= d2d_nodes_distances_table[i][j] and i!=j:
                    smallest_distance['table_position'] = (i, j)
                    smallest_distance['distance'] = d2d_nodes_distances_table[i][j]
        x = smallest_distance['table_position'][0]
        y = smallest_distance['table_position'][1]
        d2d_pairs_table[f'D2D_LINK:{d2d_pairs_index}'] = \
            ([f'{d2d_nodes[x].id}',
             f'{d2d_nodes[y].id}'], smallest_distance['distance'])
        d2d_nodes[x].set_link_id(f'D2D_LINK:{d2d_pairs_index}')
        d2d_nodes[y].set_link_id(f'D2D_LINK:{d2d_pairs_index}')
        it_index.pop(it_index.index(x))
        it_index.pop(it_index.index(y))        
        d2d_pairs_index = d2d_pairs_index+1
        smallest_distance = {'table_position': (99, 99), 'distance': 1e6}   
    for i in d2d_pairs_table.keys():
        d2d_pairs_pathloss_table[i] = \
            channel.calculate_pathloss(d2d_pairs_table[i][1])
    return d2d_pairs_table, d2d_pairs_pathloss_table


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def jain_index(vec: List[float]):
    return np.sum(vec) ** 2 / (len(vec)*np.sum([v ** 2 for v in vec]))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
