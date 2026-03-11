# ⚡ Photon ⚡

**P**arallel **H**ardware **O**ptimized **T**ensor **O**perating **N**etwork

## Description

A Deep Learning framework with an imperative style API, similar to PyTorch.

Implements a dynamic computational graph in Python that dispatches heavy arithmetic operations to
an optimized C++/CUDA backend.

## Status

Building initial gpu backend, key kernels, profiling. 

- Need more robust linking as the gpu library was using cpu implementations as they were already compiled first. For now i just swapped the static linking order so I can focus on getting the GPU version working. 

- Need new common directory for the identical metadata based operations for cpu/gpu backend. 

## Build

- Note: need g++12 and nvcc 12.8 for compatibility with blackwell architecture on ubuntu 
