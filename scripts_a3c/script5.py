import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp


def train(net, q):
    data = torch.randn((5, 2))
    x = net(data)
    print(x)
    q.put('success')


def test(q):
    q.put('hello')


def cuda_train(net, q):
    data = q.get()
    x = net(data)
    print(x)
    q.put('success')


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x = self.l1(input)
        x = F.relu(x)
        x = self.l2(x)
        output = F.softmax(x, dim=1)
        return output


if __name__ == '__main__':
    mp.set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # nn parameters
    input_size = 2
    output_size = 2
    hidden_size = 8
    # multiprocessing parameters
    num_processes = 2
    net = Model(input_size, output_size, hidden_size).to(device)
    net.share_memory()
    q = mp.Queue()
    q.put(torch.randn((5, 2)).to(device))
    p = mp.Process(target=cuda_train, args=(net, q))
    # p = mp.Process(target=test, args=(q,))
    p.start()
    # print(q.get())
    p.join()
