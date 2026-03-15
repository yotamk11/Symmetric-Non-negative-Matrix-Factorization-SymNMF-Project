#ifndef SYMNMF_H
#define SYMNMF_H

/* Function Prototypes */
double** allocate_matrix(int rows, int cols);
void free_matrix(double **matrix, int rows);
void compute_sym(double **X, double **A, int n, int d);
void compute_ddg(double **A, double **D, int n);
void compute_norm(double **A, double **D, double **W, int n);
void symnmf_algorithm(double **W, double **H, int n, int k);

#endif