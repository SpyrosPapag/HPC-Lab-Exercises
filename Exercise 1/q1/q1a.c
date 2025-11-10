#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void MPI_Exscan_pt2pt(const int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm);

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int indata = 10*(rank+1);
    int outdata = 0;

    MPI_Exscan_pt2pt(&indata, &outdata, 1, MPI_SUM, MPI_COMM_WORLD);
    printf("process %d: %d -> %d\n", rank, indata, outdata); 

    MPI_Finalize();
    return 0;
}


void MPI_Exscan_pt2pt(const int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm)
{

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int *tempbuf = (int *)malloc(count * sizeof(int));
    int *partial = (int *)malloc(count * sizeof(int));

    //se periptwsh poy den exei arxikopoihthei se 0 h se 1 (oudeteres times gia sum kai prod)
    for (int i = 0; i < count; i++) 
    {
        recvbuf[i] = 0;
    }

    // indata
    for (int i = 0; i < count; i++) 
    {
        partial[i] = sendbuf[i];
    }

    if (rank > 0) {

        MPI_Recv(tempbuf, count, MPI_INT, rank - 1, 0, comm, MPI_STATUS_IGNORE);

        for (int i = 0; i < count; i++) 
        {
            recvbuf[i] = tempbuf[i];  
            if (op == MPI_SUM)
                partial[i] += tempbuf[i];
            else if (op == MPI_PROD)
                partial[i] *= tempbuf[i];
        }
    }

    if (rank < size - 1) 
        MPI_Send(partial, count, MPI_INT, rank + 1, 0, comm);

    free(tempbuf);
    free(partial);
}

