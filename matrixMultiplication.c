#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define N 10000

void fill_matrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)(rand() % 100);
    }
}

void serial_matrix_multiply(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int compare_matrix(double* A, double* B, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(A[i] - B[i]) > 1e-6) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    int rank, size;
    double *A = NULL, *B = NULL, *C = NULL, *serial_C = NULL;
    double *local_A, *local_C;
    int rows_per_proc;
    double start, end, parallel_time = 0, serial_time = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rows_per_proc = N / size;

    local_A = (double*)malloc(rows_per_proc * N * sizeof(double));
    local_C = (double*)malloc(rows_per_proc * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));

    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));
        serial_C = (double*)malloc(N * N * sizeof(double));

        srand(time(NULL));
        fill_matrix(A, N, N);
        fill_matrix(B, N, N);
    }

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    parallel_time = end - start;

    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        start = MPI_Wtime();
        serial_matrix_multiply(A, B, serial_C, N);
        end = MPI_Wtime();
        serial_time = end - start;

        int correct = compare_matrix(C, serial_C, N);
        printf("Matrix multiplication result is %s\n", correct ? "CORRECT" : "INCORRECT");
        printf("Parallel time: %.2f seconds\n", parallel_time);
        printf("Serial time: %.2f seconds\n", serial_time);
        printf("Speedup: %.2fx\n", serial_time / parallel_time);
    }

    free(local_A); free(local_C); free(B);
    if (rank == 0) {
        free(A); free(C); free(serial_C);
    }

    MPI_Finalize();
    return 0;
}
