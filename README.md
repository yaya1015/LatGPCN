# Latent-Graph Progressive Convolution
This is a PyTorch implementation of "Latent-Graph Progressive Convolution for Robust Graph Learning".

## Requirements
The codebase is implemented in Python 3.6.8. package versions used for development are just below
- pytorch=1.9.0
- numpy=1.19.0
- deeprobust=0.2.4
- scipy=1.5.4
- tqdm=4.62.3
- torch-cluster=1.5.9
- torch-geometric=1.7.2
- torch-scatter=2.0.8
- torch-sparse=0.6.11
- torchvision=0.10.0

## Usage
Run LatGPCN model on all five datasets under Metattack using:

```python run_meta.py```

The finall results are recorded in "result.txt".

Run LatGPCN model on all five datasets under Random attack using:

```python run_random.py```

The finall results are recorded in "result.txt".

## Training process
We provide the training process output records of cora and citeseer datasets on 0.25% mettack on file "cora_meta_2.5.txt" and "citeseer_meta_2.5.txt" respectively.

## References

[1] [Jin & Ma, Graph Structure Learning for Robust Graph Neural Networks, 2020](https://dl.acm.org/doi/10.1145/3394486.3403049)

