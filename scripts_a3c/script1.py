import torch
from torch import optim, nn
from torch.functional import F
import torch.multiprocessing as mp


def data_func(net, device, train_queue):
    for i in range(5):
        sample = torch.randn((num_samples, input_size)).to(device)
        answer = net(sample)
        train_queue.put(answer)
    pass


if __name__ == '__main__':
    # multiprocess mode
    mp.set_start_method('spawn')
    # nn parameters
    input_size = 2
    output_size = 2
    hidden_size = 8
    num_samples = 5
    batch_size = 1024
    # multiprocessing parameters
    processes_count = 2
    # torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # nn
    net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.Softmax(dim=1)
    ).to(device)
    # share the nn weights accross processes
    net.share_memory()
    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    # samples and labels
    samples = torch.randn((num_samples, input_size)).to(device)
    labels = torch.randint(0, 2, (num_samples, 1))
    # multiprocess structures
    train_queue = mp.Queue(maxsize=processes_count)
    data_proc_list = []
    # multiprocess execution
    for _ in range(processes_count):
        data_proc = mp.Process(
            target=data_func,
            args=(net, device, train_queue)
        )
        data_proc.start()
        data_proc_list.append(data_proc)
    results = train_queue.get()
    print(results)
