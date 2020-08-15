import sys
import ntpath
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

# def path_leaf(path):
#     head, tail = ntpath.split(path)
#     return tail or ntpath.basename(head)

# filename = path_leaf(__file__)
# filename = filename.split('.')[0]
# print(filename)

d1 = range(10)
d2 = range(10)
ld1 = len(d1)
ld2 = len(d2)

def foo(x, y):
    res = 10*x + y
    return res

count = 0
loss = 0
for i in d1:
    for j in d2:
        x = foo(i,j)
        print(f'{count}, {i}, {j}, {x}')
        loss += abs(x - count)
        count += 1

print(loss)