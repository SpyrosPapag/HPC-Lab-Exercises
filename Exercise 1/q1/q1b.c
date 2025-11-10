#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void MPI_Exscan_omp(const int *sendbuf, int *recvbuf, MPI_Comm comm, int *offset);

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int num_threads = 2;
    int indata[num_threads];
    int offset = 0; //to offset den einai to offset se arxeio alla to meriko athroisma toy prwtoy thread toy epomenoy process 

    #pragma omp parallel num_threads(num_threads) shared(indata, offset)
    {
        int id = omp_get_thread_num();
        indata[id] = 10 * (rank * num_threads + id + 1);
        int outdata = 0;
        #pragma omp barrier // sigoureuw oti to indata[id] exei parei oles tis times
    
        MPI_Exscan_omp(indata, &outdata, MPI_COMM_WORLD, &offset);
        #pragma omp critical
        {
            printf("MPI process %d: Thread %d, indata=%d, outdata=%d\n", rank, id, indata[id], outdata);
        } 
    }

    MPI_Finalize();
    return 0;
}


void MPI_Exscan_omp(const int *sendbuf, int *recvbuf, MPI_Comm comm, int *offset)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int id = omp_get_thread_num();

    //se periptwsh poy den exei arxikopoihthei se 0
    *recvbuf = 0;

    if (rank > 0) 
    {
        int first_thread = 0; //tha mporousa kai pragma omp single alla thelo explicit
        if (id == first_thread) 
            MPI_Recv(offset, 1, MPI_INT, rank - 1, 0, comm, MPI_STATUS_IGNORE);
        #pragma omp barrier
    }

    for (int i = 0; i < id; i++) 
    {
        *recvbuf += sendbuf[i];
    }

    //prosthetw to offset to opoio einai 0 gia rank = 0
    *recvbuf += *offset;

    if (rank < size - 1) 
    {
        int last_thread = omp_get_num_threads() - 1;
        if (id == last_thread) 
        {
            int offset_value = *recvbuf + sendbuf[last_thread];
            MPI_Send(&offset_value, 1, MPI_INT, rank + 1, 0, comm);
        }
    }
}