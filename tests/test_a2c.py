from sys_simulator.a2c.a2c import compute_gae_returns
import torch


# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_gae():
    # init values and rewards
    values_n = [1, -1, 0, 2]
    rewards_n = [1, 2, -1]
    N = 3
    GAMMA = 1
    LAMBDA = 1
    values = []
    rewards = []
    # create tensor structure
    for n in range(N):
        values.append(values_n)
        rewards.append(rewards_n)
    values = torch.tensor(values).to(device)
    rewards = torch.tensor(rewards).to(device)
    # compute gae
    advantages, _ = compute_gae_returns(device, rewards, values,
                                        GAMMA, LAMBDA)
    # correct answer
    ans = []
    for n in range(N):
        ans.append([3., 4., 1.])
    ans = torch.tensor(ans).to(device)
    # normalization
    # for i in range(ans.shape[0]):
    #     ans[i] = (ans[i] - torch.mean(ans[i])) / \
    #                     (torch.std(ans[i]) + 1e-9)
    # check it is correct
    assert torch.all(ans == advantages)
