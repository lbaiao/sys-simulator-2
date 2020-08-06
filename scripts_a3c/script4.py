from multiprocessing import Process, Lock


def f(lk, i):
    lk.acquire()
    try:
        print('hello world', i)
    finally:
        lk.release()


if __name__ == '__main__':
    lock = Lock()
    for num in range(10):
        Process(target=f, args=(lock, num)).start()
