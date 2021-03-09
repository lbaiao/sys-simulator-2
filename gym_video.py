from scripts_gym.script2 import test_video
import torch

NUM_EPISODES = 10
STEPS_PER_EPISODE = 100

framework = torch.load("D:\Dev\sys-simulator-2\models\ddpg\gym\script2\\20210309-161459\\framework.pt")  # noqa
test_video(framework, NUM_EPISODES, STEPS_PER_EPISODE)
