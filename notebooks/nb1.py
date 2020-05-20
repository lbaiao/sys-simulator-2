# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline


# %%
filepath = 'D:\Dev\sys-simulator-2\data\script2_9.pickle'
file = open(filepath, 'rb')
data = pickle.load(file)


# %%
# for (i, c) in zip([2,5,10], data['action_counts_total']):
#     plt.figure()
#     plt.plot(np.mean(c, axis=0), '*',label='mean')
#     plt.plot(np.var(c, axis=0), 'x', label='variance')
#     plt.legend()
#     plt.title(f'Actions Frequencies, N={i}')

# plt.show()


# %%
d2d_speffs_avg_total = data['d2d_speffs_avg_total']
mue_success_avg_total = data['mue_success_avg_total']

fig2, ax1 = plt.subplots(figsize=(7,5))
ax1.set_xlabel('Number of D2D pairs in the RB', fontsize='x-large', )
ax1.set_ylabel('D2D Average Spectral Efficiency [bps/Hz]', color='tab:blue', fontsize='x-large')
ax1.plot(d2d_speffs_avg_total, '.', color='tab:blue', markersize=10)
ax1.tick_params(axis='both', labelsize=12)

ax2 = ax1.twinx()
ax2.set_ylabel('MUE Success Rate', color='tab:red', fontsize='x-large')
ax2.plot(mue_success_avg_total, '.', color='tab:red', markersize=10)
ax2.tick_params(axis='both', labelsize=12)
fig2.tight_layout()

# fig2, ax1 = plt.subplots()
# ax1.set_xlabel('Number of D2D pairs in the RB')
# ax1.set_ylabel('D2D Average Spectral Efficiency [bps/Hz]', color='tab:blue')
# ax1.plot(d2d_speffs_avg_total, '.', color='tab:blue')

# ax2 = ax1.twinx()
# ax2.set_ylabel('MUE Success Rate', color='tab:red')
# ax2.plot(mue_success_avg_total, '.', color='tab:red')
# fig2.tight_layout()

plt.show()

