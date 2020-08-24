from sys_simulator.channels import BANChannel


def test_ban_channel():
    channel1 = BANChannel(env='office')
    channel2 = BANChannel(env='ferry')
    d = 10
    _, loss1, _, _, _ = channel1.step(d)
    _, loss2, _, _, _ = channel2.step(d)
    loss1_mag = 10**(loss1/10)
    loss2_mag = 10**(loss2/10)
    assert (loss1, loss1_mag) == (49.1, 81283.05161640995)
    assert (loss2, loss2_mag) == (42.099999999999994, 16218.100973589266)
