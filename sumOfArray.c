#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_SIZE 1000000

int main(int argc, char** argv) {
    int rank, size;
    int *data = NULL;
    int local_size;
    int *local_data;
    long long local_sum = 0, total_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        data = (int*)malloc(TOTAL_SIZE * sizeof(int));
        for (int i = 0; i < TOTAL_SIZE; ++i) {
            data[i] = i + 1;
        }
    }

    local_size = TOTAL_SIZE / size;
    local_data = (int*)malloc(local_size * sizeof(int));

    double start_time = MPI_Wtime();

    MPI_Scatter(data, local_size, MPI_INT, local_data, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_size; ++i) {
        local_sum += local_data[i];
    }

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Total sum = %lld\n", total_sum);
        printf("Wall time = %f seconds\n", end_time - start_time);
        free(data);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}

