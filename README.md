# Synapse
Synapse is a high performance neural network engine built from first principles in C++ and Rust, featuring custom tensors, autograd, GPU acceleration, optimized runtime execution, and scalable infrastructure for real time AI and multi agent systems.

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```
