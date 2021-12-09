This directory contains the source code of defense experiment (Section 5) we included in the paper:

* The [results](https://github.com/pasta41/deception/tree/main/src/defense/results) repo contains our logs with 125 entries over three optimizers: Heavy Ball SGD, Nesterov SGD and Adam.
* The [main.py](https://github.com/pasta41/deception/blob/main/src/defense/main.py) file contains the source of of producing the logs. One can launch that by ```bash run.sh```.
* The [conclude.py](https://github.com/pasta41/deception/blob/main/src/defense/conclude.py) files gives the conclusion based on our defense protocol (Algorithm 1).

For more details on the defense experiment, please refer to our paper [Hyperparameter Optimization Is Deceiving Us, and How to Stop It](https://arxiv.org/pdf/2102.03034.pdf). 
