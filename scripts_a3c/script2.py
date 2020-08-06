import torch
from torch import optim, nn
from torch.functional import F
import torch.multiprocessing as mp


def train(net):
    # Construct data_loader, optimizer, etc.
    for i in range(5):
        data = torch.randn((num_samples, input_size)).to(device)
        labels = torch.randint(0, 2, (num_samples, 1))
        optimizer.zero_grad()
        F.mse_loss(net(data), labels).backward()
        optimizer.step()  # This will update the shared parameters


if __name__ == '__main__':
    # # multiprocess mode
    # mp.set_start_method('spawn')
    # torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # nn parameters
    input_size = 2
    output_size = 2
    hidden_size = 8
    num_samples = 5
    batch_size = 1024
    # multiprocessing parameters
    num_processes = 2
    net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.Softmax(dim=1)
    ).to(device)
    # NOTE: this is required for the ``fork`` method to work
    net.share_memory()
    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(net,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
