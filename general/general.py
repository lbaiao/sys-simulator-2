import numpy as np
import random
import math

def bits_gen(n):
    return [random.randint(0,1) for b in range(1,n+1)]

def db_to_power(x):
    return 10**(x/10)

def upsample(input, factor):
    z_mat = np.zeros([factor-1, len(input[0])])
    aux = np.concatenate((input, z_mat), axis=0)
    aux2 = np.transpose(aux)
    output = np.reshape(aux2, (1, len(input[0])*factor))
    return output

def downsample(input, factor):
    output = []
    for i in range(0, len(input)):
        if i%factor == 0:
            output.append(input[i])           
    return output

def ber(tx_signal, rx_signal):
    return np.sum(np.abs(tx_signal - rx_signal))/len(tx_signal)

def bpsk_theoric(snr):
    #snr in dB    
    snr_mag = [10**(x/10) for x in snr]        
    return [0.5*math.erfc(np.sqrt(i)) for i in snr_mag]

def distribute_users(mobile_users, d2d_users, base_station):
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

