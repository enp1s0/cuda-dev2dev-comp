# CUDA device to device memcpy comparison

`cudaMemcpy` vs copy kernel

## Run
```
git clone https://github.com/enp1s0/cuda-dev2dev-comp
cd cuda-dev2dev-comp
make
./d2d-comp.test
```

## Result example

- NVIDIA A100 (40GB SXM4)

![result-a100](./imgs/result-a100.png)

- NVIDIA V100 (16GB PCIe)

![result-v100](./imgs/result-v100.png)

- NVIDIA GeForce RTX3080 (10GB)

![result-rtx3080](./imgs/result-rtx3080.png)

## LICENSE

MIT
