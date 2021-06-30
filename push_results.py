import os


os.system('git add data/*')
os.system('git add data/a2c/*')
os.system('git add models/*')
os.system('git add models/a2c/*')
os.system('git add logs/*')
os.system('git add logs/a2c/*')
os.system('git commit -m "new results"')
os.system('git push origin devel')
