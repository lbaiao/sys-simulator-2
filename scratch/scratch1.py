import sys
# sys.path.insert(1, 'D:\\Dev\\sys-simulator-2')

import os
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from devices.devices import d2d_user, d2d_node_type

x = d2d_user(1, d2d_node_type.TX)

print(x)

print('sucesso')