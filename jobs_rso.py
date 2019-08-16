import os
import time

jname = "rso"
ntasks = 20
time_ = "24:00:00"

f = open(jname + ".slurm", "w")
f.write("#!/bin/bash \n")
f.write("#SBATCH --ntasks={}\n".format(ntasks))
f.write("#SBATCH --time={}\n".format(time_))
f.write("#SBATCH --output=" + jname + ".txt" + "\n")
f.write("#SBATCH --error=" + "e" + jname + ".txt" + "\n")
f.write("#SBATCH --job-name=" + jname + "\n")
f.write("cd /home/rcf-proj2/ma3/azizim/RSO/" + "\n")
f.write("source /usr/usc/python/3.6.0/setup.sh" + "\n")
f.write("python3 all_models.py ")
print(jname)
f.close()
time.sleep(2)
os.system("sbatch "+jname + ".slurm")
