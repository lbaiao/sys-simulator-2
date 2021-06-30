from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
import numpy as np


def test_ban_channel():
    channel1 = BANChannel(env='office', rnd=False)
    channel2 = BANChannel(env='ferry', rnd=False)
    d = 10
    loss1 = channel1.step(d)
    loss2 = channel2.step(d)
    loss1_mag = 10**(loss1/10)
    loss2_mag = 10**(loss2/10)
    assert (loss1, loss1_mag) == (49.1, 81283.05161640995)
    assert (loss2, loss2_mag) == (42.099999999999994, 16218.100973589266)


def test_winner_channel():
    carrier_frequency = 2.4  # carrier frequency in GHz
    bs_height = 25  # BS antenna height in m
    device_height = 1.5  # mobile devices height in m
    channel_to_bs = UrbanMacroNLOSWinnerChannel(
        rnd=True, f_c=carrier_frequency, h_bs=bs_height, h_ms=device_height
    )
    # np.random.seed(42)
    x1 = channel_to_bs.step(500)
    np.random.seed(42)
    x2 = channel_to_bs.step(500)
    x3 = channel_to_bs.step(500)
    print(x1)
    print(x2)
    print(x3)


if __name__ == '__main__':
    np.random.seed(42)
    test_winner_channel()
