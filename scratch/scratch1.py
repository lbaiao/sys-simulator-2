import sys
sys.path.insert(1, 'D:\\Dev\\sys-simulator-2')

from devices.devices import d2d_user, d2d_node_type

x = d2d_user(1, d2d_node_type.TX)

print(x)

print('sucesso')