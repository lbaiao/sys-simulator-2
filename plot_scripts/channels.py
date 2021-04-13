from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


sns.set_style("darkgrid")
# sns.set_palette("viridis")
# sns.set_palette("rocket")
# sns.set_theme(style="whitegrid")
# N_SAMPLES = int(1e5)
# N_BINS = int(2e2)
N_SAMPLES = int(1e3)
N_BINS = int(1e2)
d = np.linspace(1e-9, 500, N_SAMPLES)
ban_channel = BANChannel()
urban_channel = UrbanMacroNLOSWinnerChannel(sigma=2.0)
# get channel data
ban_pathlosses = ban_channel.pathloss(d)
ban_large_scale = [ban_channel.large_scale() for _ in range(N_SAMPLES)]
ban_small_scale = [ban_channel.small_scale() for _ in range(N_SAMPLES)]
urban_pathlosses = urban_channel.pathloss(d)
urban_large_scale = [urban_channel.large_scale() for _ in range(N_SAMPLES)]
urban_small_scale = [urban_channel.small_scale() for _ in range(N_SAMPLES)]
# data
ban_dict = {
    'channel': ['BAN' for _ in range(N_SAMPLES)],
    'pathloss': ban_pathlosses,
    'large_scale': ban_large_scale,
    'small_scale': ban_small_scale,
    'distances': d,
}
urban_dict = {
    'channel': ['URBAN' for _ in range(N_SAMPLES)],
    'pathloss': urban_pathlosses,
    'large_scale': urban_large_scale,
    'small_scale': urban_small_scale,
    'distances': d,
}
df = pd.DataFrame.from_dict(ban_dict)
df = df.append(pd.DataFrame.from_dict(urban_dict))
# pathlosses fig
plt.figure()
sns.lineplot(
    data=df,
    hue='channel',
    x='distances',
    y='pathloss',
)
plt.xlabel('Loss [dB]')
# plt.savefig('/home/lucas/dissertacao/figs/channel_pathlosses.eps')
# shadowings fig
plt.figure()
sns.kdeplot(
    data=df,
    hue='channel',
    x='large_scale',
    # multiple='stack',
)
plt.xlabel('Loss [dB]')
# plt.savefig('/home/lucas/dissertacao/figs/channel_shadowings.eps')
# small scale fig
plt.figure()
sns.kdeplot(
    data=df,
    hue='channel',
    x='small_scale',
)
plt.xlabel('Loss [dB]')
# plt.savefig('/home/lucas/dissertacao/figs/channel_small_scale_fadings.eps')
plt.show()
