# heinsen_routing_2022_paper

Instructions for replicating the results in "An Algorithm for Routing Vectors in Sequences" (Heinsen, 2022).


## Replication of results in paper

We recommend recreating our setup in a virtual Python environment, with the same versions of all libraries and dependencies. Runing the code requires at least one Nvidia GPU with 11GB+ RAM, along with a recent working installation of CUDA. The code is meant to be easily modifiable to work with more GPUs, or with TPUs. It is also meant to be easily modifiable to work with frameworks other than PyTorch (as long as they support Einsten summation notation for describing multilinear operations), such as TensorFlow. To replicate our environment and results, follow these steps:

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
```

5. Train all models.

You may have to modify the code a bit to work with your setup. Depending on your hardware, it could take a few hours to a few days to train all models:

```
.\train_all_models.sh
```

6. Open the Jupyter notebooks:

Make sure the virtual environment is activated


## Pretrained weights

We have made pretrained weights available:

```python
import torch
```


## Notes

We have tested our code only on Ubuntu Linux 20.04 with Python 3.8+.


## Citing

If our work is helpful to your research, please cite it:

```
@misc{heinsen2019algorithm,
    title={An Algorithm for Vectors in Sequences},
    author={Franz A. Heinsen},
    year={2022},
    eprint={XXXX.XXXXX},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## How is this used at GlassRoom?

We conceived and implemented this routing algorithm to be a component (i.e., a layer) of larger models that are in turn part of our AI software, nicknamed Graham. Our implementation of the algorithm is designed to be plugged into or tacked onto existing PyTorch models with minimal hassle. Most of the original work we do at GlassRoom tends to be either proprietary in nature or tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code and release it as stand-alone open-source software without having to disclose any key intellectual property.

We hope others find our work and our code useful.
