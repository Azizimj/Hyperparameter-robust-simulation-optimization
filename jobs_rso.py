import os
import time

epochs = 15
max_eval_hpopt = 70

model_name = "nn"
# model_name = "svm"

side_task = "none"
# side_task = 'tr_tes_sep'
# side_task = 'divide_file'
# side_task = 'sample_folder_build'
side_task = 'hyperopt_use'

jname = "rso"+str(epochs)+model_name
ntasks = 400
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
f.write("python3 all_models.py "+str(epochs)+" "+side_task+ " "+ str(max_eval_hpopt))
print(jname)
f.close()
time.sleep(2)
os.system("sbatch "+jname + ".slurm")
