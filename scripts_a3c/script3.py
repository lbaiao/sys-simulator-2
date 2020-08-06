import multiprocessing as mp


def foo(q):
    q.put('hello')
    q.put('hello2')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    print(q.get())
    p.join()
