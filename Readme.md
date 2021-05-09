
# Day and Night
See Readme From https://github.com/cameronfabbri/Day-Night-Classification

# MNIST 
Set mnist_on on in RSO.py
1. Run mylhs.py to get the LHS design points for a given range of Hypes
2. Run RSO.py with RSO_use flag on to get RSO results into res/mnist_rso.csv
3. Run RSO.py with hyperopt_use flag on to get HP result
4. Run RSO.py with hype_given flag on to get final results of RSO vs HP.

We don't need to make data folder etc, directly run RSO.py with desired flags, so set 
1. hyperopt_use=1 to use HyperOpt library.
2. RSO_use=1 to get the experimental points results for RSO.
3. hype_given=1 to get the test accuracies for a given Hyperparameter 
   (like output of RSO) on different noise levels.


