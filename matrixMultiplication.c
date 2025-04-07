#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 10000  
#define dtype double

void fill_matrix(dtype* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (dtype)(rand() % 100);
    }
}

void matrix_multiply(dtype* local_A, dtype* B, dtype* local_C, int rows, int n) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) {
            dtype sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    dtype *A = NULL, *B = NULL, *C = NULL;
    dtype *local_A, *local_C;
    int rows_per_proc;
    double start, end, parallel_time = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            printf("Matrix size %d is not divisible by number of processes %d.\n", N, size);
        }
        MPI_Finalize();
        return 1;
    }

    rows_per_proc = N / size;

    local_A = (dtype*)malloc(rows_per_proc * N * sizeof(dtype));
    local_C = (dtype*)malloc(rows_per_proc * N * sizeof(dtype));
    B = (dtype*)malloc(N * N * sizeof(dtype));

    if (rank == 0) {
        A = (dtype*)malloc(N * N * sizeof(dtype));
        C = (dtype*)malloc(N * N * sizeof(dtype));
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

    matrix_multiply(local_A, B, local_C, rows_per_proc, N);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    parallel_time = end - start;

    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Parallel multiplication complete with %d processes.\n", size);
        printf("Parallel time: %.2f seconds\n", parallel_time);
    }

    free(local_A);
    free(local_C);
    free(B);
    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
