import sys
import numpy as np
import pandas as pd
import symnmf

def print_matrix(matrix):
    """Prints the matrix with 4 decimal places
    Args:
        matrix: 2D list to be printed."""
    
    for row in matrix:
        print(",".join(format(val, ".4f") for val in row))

def main():
    """Main function to execute SymNMF operations based on command-line arguments.
    Usage: python symnmf.py <k> <goal> <input_file>
    where:
        k: The number of clusters (positive integer).
        goal: The operation to perform ("sym", "ddg", "norm", or "symnmf").
        input_file: The file containing the data points."""
    
    # 1. Parse arguments (Section 2.1)
    if len(sys.argv) < 4:
        raise ValueError("Invalid number of arguments")
    k = int(sys.argv[1])
    goal = sys.argv[2]
    file_name = sys.argv[3]
    np.random.seed(1234)
    
    # 2. Load data from file
    data = pd.read_csv(file_name, header=None).values.tolist()
    n = len(data)
    if (k > n or k <= 0):
        raise ValueError("Invalid number of clusters")
    d = len(data[0])
    # 3. Execution logic based on goal (Section 2.1)
    if goal == "sym":
        result = symnmf.sym(data, n, d)
        print_matrix(result)
        
    elif goal == "ddg":
        result = symnmf.ddg(data, n, d)
        print_matrix(result)
        
    elif goal == "norm":
        result = symnmf.norm(data, n, d)
        print_matrix(result)
        
    elif goal == "symnmf":
        W = symnmf.norm(data, n, d)
        m = np.mean(W)
        
        upper_bound = 2 * np.sqrt(m / k)
        initial_H = np.random.uniform(0, upper_bound, (n, k)).tolist()
        final_H = symnmf.symnmf(W, initial_H, n, k)
        print_matrix(final_H)
    
    else:
        raise ValueError("Invalid goal")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)