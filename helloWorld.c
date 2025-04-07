#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int cluster_size;

    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);

    int pi_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &pi_num);

    printf("Hello World from process %d out of %d processes\n", pi_num, cluster_size);

    MPI_Finalize();
    return 0;
}