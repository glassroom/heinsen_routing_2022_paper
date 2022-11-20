# heinsen_routing_2022_paper

Instructions for replicating the results in "An Algorithm for Routing Vectors in Sequences" (Heinsen, 2022).


## Replication of Results in Paper

We recommend recreating our setup in a virtual Python environment, with the same versions of all libraries and dependencies. To replicate our environment and results, follow these steps:

1. Change to the directory in which you cloned this repository:

```
cd /home/<my_name>/<my_directory>
```

2. Create a new Python 3 virtual environment:

```
virtualenv --python=python3 python
```

3. Activate the virtual environment:

```
source ./python/bin/activate
```

4. Install required Python libraries in environment:

```
pip install --upgrade pip
pip install --upgrade -r requirements.txt
pip install git+https://github.com/glassroom/torch_train_test_loop
pip install git+https://github.com/glassroom/heinsen_routing
```

5. Open the Jupyter notebooks:

There are two notebooks with sample code for computing and dispaying end-to-end credit assignments. One notebook is for natural language tasks. The other one is for visual tasks. Start Jupyter Lab to open and run them:

```
jupyter lab
```


## Training All Classification Heads from Scratch

We provide code for training all classification heads. As written, the training code assumes you have at least one Nvidia GPU with 24GB+ RAM along with a recent working installation of CUDA, but the code is meant to be easily modifiable to work with different setups. Follow these steps to train all classification heads:

1. Download the ImageNet-1K dataset to `.data/vision/imagenet/`. Create the directory if necessary.

2. Review the training code in `train.py` and modify it as necessary so it works with your particular hardware and software configuration.

3. Review the shell script `train_all_heads.sh` and modify it as necessary so it works with your particular hardware and software configuration (e.g., you can decrease batch sizes while increasing number of iterations per epoch to reduce memory consumption).

4. Make sure the virtual environment is activated (`source ./python/bin/activate`), and then run the shell script to train all classification heads:

```
./train_all_heads.sh
```

Depending on your setup, training all heads may take as little as a few hours or up to a few days.


## Notes

We have tested the code in this repository only on Ubuntu Linux 20.04 with Python 3.8+.


## Citing

If our work is helpful to your research, please cite it:

```
@misc{heinsen2020algorithm,
    title={An Algorithm for Vectors in Sequences},
    author={Franz A. Heinsen},
    year={2022},
    eprint={PENDING},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## How is this used at GlassRoom?

We conceived and implemented this routing algorithm to be a component (i.e., a layer) of larger models that are in turn part of our AI software, nicknamed Graham. Most of the original work we do at GlassRoom tends to be either proprietary in nature or tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code and release it as stand-alone open-source software without having to disclose any key intellectual property. We hope others find our work and our code useful.
