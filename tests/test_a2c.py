from a2c.a2c import compute_gae_returns
import torch


def test_gae():
    # init values and rewards
    values_n = [1, -1, 0, 2]
    rewards_n = [1, 2, -1]
    N = 3
    values = []
    rewards = []
    # create tensor structure
    for n in N:
        values.append(values_n)
        rewards.append(rewards_n)
    values = torch.tensor(values)
    rewards = torch.tensor(rewards)
    # compute gae
    advantages = compute_gae_returns(rewards, values)
    # correct answer
    ans = []
    for n in N:
        ans.append([3, 4, 1])
    ans = torch.tensor(ans)
    # check it is correct
    equal_values = ans == advantages
    assert equal_values.mean() == 1
