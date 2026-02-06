# IMEXLBM: GPU-Accelerated Lattice Boltzmann Solver

**IMEXLBM** is a high-performance, single-phase **Lattice Boltzmann Method (LBM)** solver engineered for next-generation GPU-accelerated architectures. By leveraging the **Kokkos** performance portability library, IMEXLBM achieves massive throughput on diverse hardware while maintaining a single codebase.

---

## 🚀 Performance & Scalability

* **GPU Acceleration:** Highly optimized kernels provide significant speedups over traditional CPU-only implementations.
* **Performance Portability:** Built on **Kokkos**, ensuring high performance across different GPU vendors (NVIDIA, AMD, Intel) and multicore CPUs.
* **Massive Parallelism:** Fully integrated with the **Message Passing Interface (MPI)** for seamless multi-node scaling on leadership-class supercomputing platforms.



## ✅ Accuracy & Validation

Precision is a core priority of IMEXLBM. Our solver is rigorously validated against canonical 2D and 3D benchmark problems. 

The results between GPU-accelerated and CPU-only platforms are **highly consistent**, with discrepancies limited strictly to floating-point round-off errors, ensuring that performance gains do not come at the cost of physical accuracy.

## 🛠️ Key Features

* **Single-Phase Flow:** High-fidelity LBM solver for incompressible fluid dynamics.
* **HPC Ready:** Designed for large-scale deployments on distributed supercomputers.
* **Cross-Platform:** Write once, run anywhere with Kokkos.

---
# IMEXLBM

> **GPU-Accelerated Lattice Boltzmann Solver with Kokkos Performance Portability**

**IMEXLBM** is a high-performance, single-phase **Lattice Boltzmann Method (LBM)** solver engineered for next-generation GPU-accelerated architectures. By leveraging the **Kokkos** performance portability library, IMEXLBM achieves massive throughput on diverse hardware while maintaining a single codebase.

---

## 🚀 Key Features

* **GPU Acceleration:** Significant speedup over traditional CPU-only implementations.
* **Performance Portability:** Powered by **Kokkos**, ensuring high performance across NVIDIA, AMD, Intel GPUs, and multicore CPUs.
* **HPC Scaling:** Fully integrated with **MPI** for multi-node scaling on leadership-class supercomputers.
* **Validated Accuracy:** Rigorous validation against canonical 2D and 3D benchmarks. Results are consistent with CPU-only calculations down to the floating-point round-off level.

---

## 🛠️ Installation

### Prerequisites
Before compiling IMEXLBM, ensure you have the following installed:
1.  **MPI:** (e.g., OpenMPI or MPICH) for distributed memory parallelism.
2.  **Kokkos:** Follow the [official Kokkos guide](https://kokkos.github.io/kokkos-core-wiki/post-install/index.html) to set up the library for your specific hardware backend.

---

## 📂 Project Structure: Lid-Driven Cavity Example

The repository includes a C++ implementation of the **Lid-Driven Cavity** problem, organized into three core modules:

1.  **Main Function:** Orchestrates the program structure and the simulation time-loop.
2.  **System:** Defines macroscopic parameters and relaxation time ($\tau$), managing the mapping between physical and simulation units.
3.  **LBM Solver:** The core engine implementing the 2D LBM streaming and collision kernels.

---

## 🏗️ Compilation Instructions

### 1. Standard C++ (CPU Debugging)
To run a basic version of the solver on a local CPU without Kokkos dependencies:

```bash
g++ ./System.cpp ./lbm.cpp ./main.cpp -o lbm_serial
