# Symmetric Non-negative Matrix Factorization (SymNMF) Project

This project implements the **Symmetric Non-negative Matrix Factorization** algorithm and compares its clustering performance against the **K-Means** algorithm using the **Silhouette Score**.

Notice: The data should be normalized for the SymNMF.

The project uses a hybrid approach:
* **Backend:** High-performance matrix calculations implemented in **C**.
* **Frontend:** Algorithm orchestration, initialization, and data visualization in **Python**.
* **Integration:** A **Python C-API** extension module connects the two.

---

## Project Structure

* `symnmf.c` / `symnmf.h`: Core C implementation of the SymNMF algorithm logic (Similarity, DDG, Normalization, and Update rules).
* `symnmfmodule.c`: The C-API wrapper that exports C functions to Python.
* `symnmf.py`: Python interface for running specific matrix goals.
* `analysis.py`: The main entry point for comparing SymNMF and K-Means.
* `kmeans.py`: Implementation of the K-Means algorithm.
* `setup.py`: Script to build and install the C extension for Python.
* `Makefile`: A script to compile the standalone C executable. Running `make` will generate the `symnmf` binary using strict flags  and link the math library.

---

## Execution Guide

### 1. Analysis Mode (`analysis.py`)
**Purpose:** The main entry point for performance evaluation.
- Executes both the full **SymNMF** and **K-Means** algorithms.
- Calculates the **Silhouette Score** for each method to measure clustering quality.
- **Output:** Prints the two scores side-by-side for comparison.

### 2. Python Interface (`symnmf.py`)
**Purpose:** Provides access to the matrix logic via Python (using the C-extension).
Supports **4 specific goals**:
- `sym`: Calculates and prints the Similarity Matrix ($A$).
- `ddg`: Calculates and prints the Diagonal Degree Matrix ($D$).
- `norm`: Calculates and prints the Normalized Similarity Matrix ($W$).
- `symnmf`: Executes the full iterative SymNMF process and prints the final matrix $H$.

### 3. Standalone C Interface (`symnmf.c`)
**Purpose:** Direct execution of core matrix logic in C, bypassing the Python layer.
Supports **3 specific goals**:
- `sym`: Calculates and prints the Similarity Matrix.
- `ddg`: Calculates and prints the Diagonal Degree Matrix.
- `norm`: Calculates and prints the Normalized Similarity Matrix.
*(Note: The full iterative update rule for SymNMF is handled via the Python module).*
