import pickle
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np

filepath = 'D:/Dev/sys-simulator-2/data/scratch20.pickle'
file = open(filepath, 'rb')
data = pickle.load(file)

MAX_NUMBER_OF_AGENTS = 10
aux_range = list(range(MAX_NUMBER_OF_AGENTS+1))[1:]

action_counts_total = data['action_counts_total']
d2d_spectral_effs = data['d2d_speffs_avg_total']
mue_success_rate = data['mue_success_rate']
equals_counts_total = data['equals_counts_total']

d2d_speffs_avg = list()
for i, d in enumerate(d2d_spectral_effs):    
    d2d_speffs_avg.append(np.average(d))

fig2, ax1 = plt.subplots(figsize=(7,5))
ax1.set_xlabel('Number of D2D pairs in the RB', fontsize='x-large', )
ax1.set_ylabel('D2D Average Spectral Efficiency [bps/Hz]', color='tab:blue', fontsize='x-large')
ax1.plot(d2d_speffs_avg, '.', color='tab:blue', markersize=10)
ax1.tick_params(axis='both', labelsize=12)

ax2 = ax1.twinx()
ax2.set_ylabel('MUE Success Rate', color='tab:red', fontsize='x-large')
ax2.plot(mue_success_rate, '.', color='tab:red', markersize=10)
ax2.tick_params(axis='both', labelsize=12)
fig2.tight_layout()


# xi = list(range(len(aux_range)))
# ax = [0,1,2,3,4]
# axi = list(range(len(ax)))
# for i, c in enumerate(action_counts_total):
#     if i in aux_range:
#         plt.figure()
#         plt.plot(np.mean(c, axis=0)/i*100, '*',label='mean')
#         plt.plot(np.std(c, axis=0)/i*100, 'x', label='std')
#         plt.legend()
#         plt.title(f'N={i}')
#         plt.xlabel('Action Index')
#         plt.ylabel('Average Action Ocurrency [%]')
#         plt.xticks(axi, ax)

# mean_equals = np.array([np.mean(c) for c in equals_counts_total])
# std_equals = np.array([np.std(c) for c in equals_counts_total])
# plt.figure()
# plt.plot(mean_equals[aux_range]*100, '*',label='mean')
# plt.plot(std_equals[aux_range]*100, 'x', label='std')
# plt.legend()
# plt.xlabel('Amount of D2D Devices')
# plt.ylabel('Average Equal Actions Ocurrency [%]')
# plt.xticks(xi, aux_range)

plt.show()