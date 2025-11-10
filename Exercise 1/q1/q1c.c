#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 2

void MPI_Exscan_omp(const int *sendbuf, int *recvbuf, MPI_Comm comm, int *offset);
void initialize_matrix(int (*matrix)[N][N], int seed);
void print_matrix(int matrix[N][N][N], int rank, int thread_id);
int equal_matrices(int matrix1[N][N][N], int matrix2[N][N][N]);


int main(int argc, char *argv[])
{
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE) 
    {
        printf("Error: MPI does not support required multithreading level.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int num_threads = 2;
    int indata[num_threads];
    int offset = 0; //to offset den einai to offset se arxeio alla to meriko athroisma toy prwtoy thread toy epomenoy process 

    int step = 420;
    char filename[256];
    sprintf(filename, "mydata_%05d.bin", step);

    MPI_File f;
    MPI_File_open(MPI_COMM_WORLD, filename , MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &f);

    MPI_File_set_size (f, 0);
    MPI_Offset base;
    MPI_File_get_position(f, &base);

    const int nlocal = N*N*N;
    int len = nlocal*sizeof(int);

    #pragma omp parallel num_threads(num_threads) shared(indata, offset)
    {
        int id = omp_get_thread_num();
        int matrix[N][N][N];
        int seed = 10 * (rank * num_threads + id + 1);
        initialize_matrix(matrix, seed);

        // #pragma omp critical
        // {
        //     print_matrix(matrix, rank, id);
        // }

        indata[id] = len;
        int fileOffset = 0;
        #pragma omp barrier // sigoureuw oti to indata[id] exei parei oles tis times
        
        MPI_Exscan_omp(indata, &fileOffset, MPI_COMM_WORLD, &offset);

        MPI_File_write_at_all(f, base + ((MPI_Offset)fileOffset), matrix, nlocal, MPI_INT, MPI_STATUS_IGNORE);
        #pragma omp barrier //sigoyreuw oti exei teleiwsei h eggrafi toy arxeio prin ksekinisw thn anagnwsh

        // elegxos swsths eggrafhs
        int read_matrix[N][N][N];
        MPI_File_read_at_all(f, base + ((MPI_Offset)fileOffset), read_matrix, nlocal, MPI_INT, MPI_STATUS_IGNORE);

        #pragma omp critical
        {
            if(!equal_matrices(read_matrix, matrix))
            {
                printf("Process: %d, Thread: %d -> Error in writing!\n", rank, id);
            } 
            else
            {
                printf("Process: %d, Thread: %d -> Correct!\n", rank, id);
            } 
        }  
    }

    MPI_File_close(&f);
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

void initialize_matrix(int (*matrix)[N][N], int seed)  
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                matrix[i][j][k] = rand_r(&seed) % 100; //0-99
            }
        }
    }
}

void print_matrix(int matrix[N][N][N], int rank, int thread_id) 
{
    printf("Matrix for Rank %d, Thread %d:\n", rank, thread_id);
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            for (int k = 0; k < N; k++) 
            {
                printf("%3d ", matrix[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

int equal_matrices(int matrix1[N][N][N], int matrix2[N][N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                if (matrix1[i][j][k] != matrix2[i][j][k])
                {
                    return 0; // not equal
                }
            }
        }
    }
    return 1; // Equal
}