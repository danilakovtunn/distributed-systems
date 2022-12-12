#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static const char* VERY_LONG_MESSAGE = "abc";

static const int REORDER = 0;
static const int NDIMS = 2;
static const int DIMS[2] = {8, 8};
static const int PERIODS[2] = {0, 0};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm_cart;
    int size;
    int rank;
    int coords[2];
    int length = strlen(VERY_LONG_MESSAGE) + 1;
    char *m = (char *) malloc(length * sizeof(char));

    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, DIMS, PERIODS, REORDER, &comm_cart);
    MPI_Comm_size(comm_cart, &size);
    MPI_Comm_rank(comm_cart, &rank);
    MPI_Cart_coords(comm_cart, rank, NDIMS, coords);

    if (rank == 0) {
        strcpy(m, VERY_LONG_MESSAGE);
    }

    if (coords[0] == 0) {
        if (coords[1] != 0) {
            MPI_Status status;
            int source_coords[2] = {0, coords[1] - 1};
            int source;
            MPI_Cart_rank(comm_cart, source_coords, &source);
            MPI_Recv(m, length, MPI_CHAR, source, length - 1, comm_cart, &status);
        }
        if (coords[1] != 7) {
            int dest;
            int dest_coords[2] = {0, coords[1] + 1};
            MPI_Cart_rank(comm_cart, dest_coords, &dest);
            MPI_Send(m, length, MPI_CHAR, dest, length - 1, comm_cart);
        }
    } else {
        MPI_Status status;
        int source_coords[2] = {coords[0] - 1, coords[1]};
        int source;
        MPI_Cart_rank(comm_cart, source_coords, &source);
        MPI_Recv(m, length, MPI_CHAR, source, length - 1, comm_cart, &status);
    }

    if (coords[0] != 7) {
        int dest;
        int dest_coords[2] = {coords[0] + 1, coords[1]};
        MPI_Cart_rank(comm_cart, dest_coords, &dest);
        MPI_Send(m, length, MPI_CHAR, dest, length - 1, comm_cart);
    }

    printf("%d: %s\n", rank, m); fflush(stdout);

    MPI_Finalize();
    return 0;
}
