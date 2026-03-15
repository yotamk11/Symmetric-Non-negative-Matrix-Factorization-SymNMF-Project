#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"
#include <string.h>

#define MAX_ITER 300
#define EPSILON 1e-4
#define beta 0.5

/* --- Memory Management and Helper Functions --- */

/**
 * Handles errors by printing an error message and exiting the program.
 */
void handle_error() {
    printf("An Error Has Occurred\n");
    exit(1);
}

/**
 * Allocates a 2D array of doubles (matrix) with given rows and columns.
 * Returns a pointer to the array of pointers (double**).
 */
double** allocate_matrix(int rows, int cols) {
    int i;
    double **matrix = (double **)malloc(rows * sizeof(double *));
    if (matrix == NULL){
        handle_error();
    }
    for (i = 0; i < rows; i++) {
        matrix[i] = (double *)malloc(cols * sizeof(double));
        if (matrix[i] == NULL){
            int j;
            for (j = 0; j < i; j++){
                free(matrix[j]);
            }
            free(matrix);
            handle_error();
        };
    }
    return matrix;
}

/**
 * Frees the memory allocated for a 2D array of doubles.
 */
void free_matrix(double **matrix, int rows) {
    int i;
    if (matrix == NULL) return;
    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/**
 * Copies the content of one matrix (src) to another (dst).
 * Both matrices must have the same dimensions.
 */
void copy_matrix(double **src, double **dst, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

/**
 * Performs general matrix multiplication: res = mat1 * mat2.
 * mat1: (r1 x c1), mat2: (c1 x c2), res: (r1 x c2).
 */
void matrix_mul(double **mat1, double **mat2, double **res, int rows1, int cols1, int cols2) {
    int i, j, l;
    for (i = 0; i < rows1; i++) {
        for (j = 0; j < cols2; j++) {
            res[i][j] = 0;
            for (l = 0; l < cols1; l++) {
                res[i][j] += mat1[i][l] * mat2[l][j];
            }
        }
    }
}

/**
 * Calculates the squared Frobenius norm of the difference between two matrices.
 * Used to check the convergence condition (Epsilon).
 */
double frobenius_norm_sq_diff(double **mat1, double **mat2, int rows, int cols) {
    int i, j;
    double sum = 0, diff;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            diff = mat1[i][j] - mat2[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

/* --- SymNMF Algorithm Implementation (Steps 1.1 - 1.4) --- */

/**
 * Step 1.1: Computes the similarity matrix A from the input data X.
 * Formula: a_ij = exp(-||xi - xj||^2 / 2) for i != j, else 0.
 * Note: works only with close points due to the exponential decay!
 * If similarity matrix is all zero, it's probable that the data points are too far apart.
 */
void compute_sym(double **X, double **A, int n, int d) {
    int i, j, k;
    double dist, diff;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) { A[i][j] = 0; continue; }
            dist = 0;
            for (k = 0; k < d; k++) {
                diff = X[i][k] - X[j][k];
                dist += diff * diff;
            }
            A[i][j] = exp(-dist / 2.0);
        }
    }
}

/**
 * Step 1.2: Computes the Diagonal Degree Matrix D.
 * Formula: d_ii = sum of row i in A.
 */
void compute_ddg(double **A, double **D, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        double sum = 0;
        for (j = 0; j < n; j++) sum += A[i][j];
        for (j = 0; j < n; j++) D[i][j] = 0;
        D[i][i] = sum;
    }
}

/**
 * Step 1.3: Computes the Normalized Similarity Matrix W. notice that D is diagonal, so we only need the diagonal elements for the computation.
 * Formula: W = D^-0.5 * A * D^-0.5.
 */
void compute_norm(double **A, double **D, double **W, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            W[i][j] = A[i][j] / (sqrt(D[i][i]) * sqrt(D[j][j]));
        }
    }
}

/**
 * Step 1.4.2: Performs a single update iteration of matrix H.
 * Formula: H = H * (0.5 + 0.5 * (WH / (HH^T)H)).
 * Uses pre-allocated auxiliary matrices to optimize performance.
 */
void update_h_step(double **W, double **H, double **WH, double **HHT, double **HHTH, int n, int k) {
    int i, j, l;
    matrix_mul(W, H, WH, n, n, k);
    
    /* Compute H * H_transpose (n x n) */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            HHT[i][j] = 0;
            for (l = 0; l < k; l++) HHT[i][j] += H[i][l] * H[j][l];
        }
    }
    
    matrix_mul(HHT, H, HHTH, n, n, k);
    
    /* Apply the update formula to each element */
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            if (HHTH[i][j] != 0) {
                H[i][j] = H[i][j] * (beta + beta * (WH[i][j] / HHTH[i][j]));
            }
        }
    }
}

/**
 * Performs the full SymNMF iterative algorithm.
 * Runs up to 300 iterations or until convergence (epsilon = 1e-4).
 */
void symnmf_algorithm(double **W, double **H, int n, int k) {
    int iter;
    double **WH = allocate_matrix(n, k);
    double **HHT = allocate_matrix(n, n);
    double **HHTH = allocate_matrix(n, k);
    double **old_H = allocate_matrix(n, k);

    for (iter = 0; iter < MAX_ITER; iter++) {
        copy_matrix(H, old_H, n, k);
        update_h_step(W, H, WH, HHT, HHTH, n, k);
        
        /* Check convergence: Frobenius norm of difference squared < 1e-4 */
        if (frobenius_norm_sq_diff(H, old_H, n, k) < EPSILON) break;
    }

    /* Free auxiliary matrices memory */
    free_matrix(WH, n);
    free_matrix(HHT, n);
    free_matrix(HHTH, n);
    free_matrix(old_H, n);
}

/* -------------------Running C file alone functions and main-----------------------------------  */


/**
 * Counts the number of points (n) and the dimension (d) in the input file.
 * Assumes the file is formatted as comma-separated values.
 */
void get_dimensions(char *filename, int *n, int *d) {
    FILE *f = fopen(filename, "r");
    char line[1024];
    *n = 0;
    *d = 0;

    if (f == NULL) return;

    while (fgets(line, sizeof(line), f)) {
        if (*n == 0) {
            char *ptr = line;
            while (*ptr) {
                if (*ptr == ',') (*d)++;
                ptr++;
            }
            (*d)++; /* Number of columns is number of commas + 1 */
        }
        (*n)++;
    }
    fclose(f);
}

/**
 * Reads the data points from the file into a pre-allocated 2D array.
 */
double** load_from_file(char *filename, int n, int d) {
    int i, j;
    FILE *f = fopen(filename, "r");
    
    double **matrix = allocate_matrix(n, d);

    if (f == NULL || matrix == NULL){
        printf("An Error Has Occurred\n");
        free_matrix(matrix, n);
        exit(1);
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            /* Read double and skip the following character (comma or newline) */
            if (fscanf(f, "%lf%*c", &matrix[i][j]) == EOF) break;
        }
    }
    fclose(f);
    return matrix;
}



/**
 * Helper to print a matrix in the required format (4 decimal places).
 */
void print_matrix(double **mat, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f%s", mat[i][j], (j == cols - 1) ? "" : ",");
        }
        printf("\n");
    }
}

/**
 * Main execution logic for the C standalone program.
 * Supports goals: sym, ddg, norm.
 */
int main(int argc, char **argv) {
    char *goal, *file_name;
    int n = 0, d = 0;
    double **X, **A, **D, **W;

    if (argc < 3){
       handle_error();
    }

    goal = argv[1];
    file_name = argv[2];

    /* Step 1: Parse dimensions and load data */
    get_dimensions(file_name, &n, &d);
    X = load_from_file(file_name, n, d);
    if (X == NULL){
        handle_error();
    }

    /* Step 2: Initialize Similarity Matrix (A) as it is required for all goals */
    A = allocate_matrix(n, n);
    compute_sym(X, A, n, d);

    /* Step 3: Execute logic based on the requested goal */
    if (strcmp(goal, "sym") == 0) {
        print_matrix(A, n, n);
    } 
    else if (strcmp(goal, "ddg") == 0) {
        D = allocate_matrix(n, n);
        compute_ddg(A, D, n);
        print_matrix(D, n, n);
        free_matrix(D, n);
    } 
    else if (strcmp(goal, "norm") == 0) {
        D = allocate_matrix(n, n);
        compute_ddg(A, D, n);
        W = allocate_matrix(n, n);
        compute_norm(A, D, W, n);
        print_matrix(W, n, n);
        free_matrix(D, n);
        free_matrix(W, n);
    }
    else{
        printf("An Error Has Occurred\n");
        free_matrix(X, n);
        free_matrix(A, n);
        exit(1);
    }

    /* Step 4: Final memory cleanup */
    free_matrix(X, n);
    free_matrix(A, n);

    return 0;
}