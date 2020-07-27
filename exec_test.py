import os
from sys_simulator import env_variables
from scripts_a2c.script6 import run as run_training
from scratch_a2c.scratch6 import run as run_test

os.environ[env_variables.MODE] = env_variables.MODE_TEST

# run training and tests
run_training()
run_test()
