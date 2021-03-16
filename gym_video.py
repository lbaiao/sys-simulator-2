# ddpg
# from scripts_gym.script2 import test_video
# ppo
# from scripts_gym.script3 import test_video
# from scripts_gym.script3_windows import test_video
# dqn
# from scripts_gym.script4 import test_video
from scripts_gym.script7_windows import test_video

import torch

NUM_EPISODES = 10
STEPS_PER_EPISODE = 100

# ddpg
# framework = torch.load("D:\Dev\sys-simulator-2\models\ddpg\gym\script2\\20210309-171628\\framework.pt")  # noqa
# ppo
# framework = torch.load("D:\Dev\sys-simulator-2\models\ppo\gym\script3\\20210313-122742\\framework.pt") # noqa
# dqn mountain-car
# framework = torch.load("D:\Dev\sys-simulator-2\models\dql\gym\script4\\20210310-093429\\framework.pt")  # noqa
# dqn cartpole
# the best, do not change
# framework = torch.load("D:\Dev\sys-simulator-2\models\dql\gym\script4\\20210312-190802\\framework.pt")  # noqa
# a2c cartpole
# the best, do not change
# framework = torch.load("D:\Dev/sys-simulator-2/models/a2c/gym/script7/20210314-155244/framework.pt") # noqa
framework = torch.load("D:\Dev/sys-simulator-2/models/a2c/gym/script7/20210314-164639/framework.pt") # noqa
test_video(framework, NUM_EPISODES, STEPS_PER_EPISODE)
