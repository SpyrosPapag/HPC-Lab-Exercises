#include <iostream>
#include <cuda.h>
#include <vector>
#include <chrono>
#include <math.h>

using namespace std;

void fill_matrix(float*, int, float);
void print_matrix(const char*, float*, int);
void cpu_matrix_subtract(float*, float*, float*, int);
void gpu_matrix_subtract(float*, float*, float*, int);
void cpu_matrix_add(float*, float*, float*, int);
void gpu_matrix_add(float*, float*, float*, int);
void cpu_matrix_multiply(float*, float*, float*, int);
void gpu_matrix_multiply(float*, float*, float*, int);
void cpu_complex_matrix_multiply(float*, float*, float*, float*, float*, float*, int);
void gpu_complex_matrix_multiply(float*, float*, float*, float*, float*, float*, int);
double get_wtime();
bool compareMatrices(float*,float*, int);

void cpuExecution(float*, float*, int, int, int);
void gpuExecution(float*, float*, int, int, int);


int main()
{
    // execution parameters
    const int singleExec = 0;
    const int maxNexp = 13; // only if singleExec = 0 & Nmax = 2^maxNexp
    const int minNexp = 10; // only if singleExec = 0 & Nmin = 2^minNexp
    const int cpuExec = 0; // only if singleExec = 1
    const int verbose = 0;
    const int checkResults = 0;

    // size of matrix
    int  N = 64;

    // declare result matrices on host
    vector<float> E(N * N);
    vector<float> F(N * N);

    if(singleExec)
    {
        if(cpuExec)
            cpuExecution(E.data(), F.data(), N, verbose, 0);
        else
            gpuExecution(E.data(), F.data(), N, verbose, 0);
        
    }
    else
    {
        for(N = pow(2,minNexp); N < pow(2, maxNexp); N*=2)
        {
            // declare result matrices for each N
            vector<float> E_loop_cpu(N * N);
            vector<float> F_loop_cpu(N * N);

            vector<float> E_loop_gpu(N * N);
            vector<float> F_loop_gpu(N * N);

            cpuExecution(E_loop_cpu.data(), F_loop_cpu.data(), N, 0, checkResults);
            gpuExecution(E_loop_gpu.data(), F_loop_gpu.data(), N, 0, checkResults);

            if(checkResults)
            {
                if(compareMatrices(E_loop_cpu.data(), E_loop_gpu.data(), N) && compareMatrices(F_loop_cpu.data(), F_loop_gpu.data(), N))
                    cout << "CPU and GPU outputs are equal for N=" << N << endl;
                else
                    cout << "CPU and GPU outputs are not equal for N=" << N << "!!!" <<endl;
            }

        }
    }
    return 0;
}

bool compareMatrices(float* A, float* B, int N)
{
    for (int i = 0; i < N * N; ++i)
    {
        if (fabs(A[i] - B[i]) > 1e-5)
            return false;
    }
    return true;
}

void gpuExecution(float *E, float *F, int N, int verbose, int checkResults)
{
    // declare matrices for device
    vector<float> d_A(N * N);
    vector<float> d_B(N * N);
    vector<float> d_C(N * N);
    vector<float> d_D(N * N);

    // initialize matrices for device
    float b_gpu = 0.0f;
    fill_matrix(d_A.data(), N, b_gpu);
    fill_matrix(d_B.data(), N, ++b_gpu);
    fill_matrix(d_C.data(), N, ++b_gpu);
    fill_matrix(d_D.data(), N, ++b_gpu);

    // do calculations
    if(!checkResults)
        cout << "running on GPU" << endl;
    double start = get_wtime();
    gpu_complex_matrix_multiply(d_A.data(), d_B.data(), d_C.data(), d_D.data(), E, F, N);
    double end = get_wtime();

    if(!checkResults)
        cout << "\tGPU finished with computation time:" << end-start << "seconds for N=" << N << endl;
    
    if(verbose)
    {
        print_matrix("A_gpu", d_A.data(), N);
        print_matrix("B_gpu", d_B.data(), N);
        print_matrix("C_gpu", d_C.data(), N);
        print_matrix("D_gpu", d_D.data(), N);
        print_matrix("E_gpu", E, N);
        print_matrix("F_gpu", F, N);
    }
}

void cpuExecution(float *E, float *F, int N, int verbose, int checkResults)
{
    // declare input matrices on host
    vector<float> A(N * N);
    vector<float> B(N * N);
    vector<float> C(N * N);
    vector<float> D(N * N);
    
    // initialize matrices on host
    float b_cpu = 0.0f;
    fill_matrix(A.data(), N, b_cpu);
    fill_matrix(B.data(), N, ++b_cpu);
    fill_matrix(C.data(), N, ++b_cpu);
    fill_matrix(D.data(), N, ++b_cpu);

    if(!checkResults)
        cout << "running on CPU" << endl;
    double start = get_wtime();
    cpu_complex_matrix_multiply(A.data(), B.data(), C.data(), D.data(), E, F, N);
    double end = get_wtime();

    if(!checkResults)
        cout << "\tCPU finished with computation time:" << end-start << "seconds for N=" << N << endl;

    if(verbose)
    {
        // Print matrices
        print_matrix("A_cpu", A.data(), N);
        print_matrix("B_cpu", B.data(), N);
        print_matrix("C_cpu", C.data(), N);
        print_matrix("D_cpu", D.data(), N);
        print_matrix("E_cpu", E, N);
        print_matrix("F_cpu", F, N);
    }
}

double get_wtime() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

void gpu_complex_matrix_multiply(float *A, float *B, float *C, float *D, float *E, float *F, int N)
{
    // temp matrices
    vector<float> AC(N * N);
    vector<float> BD(N * N);
    vector<float> AD(N * N);
    vector<float> BC(N * N);
    
    // calculate temp matrices
    gpu_matrix_multiply(A, C, AC.data(), N);
    gpu_matrix_multiply(B, D, BD.data(), N);
    gpu_matrix_multiply(A, D, AD.data(), N);
    gpu_matrix_multiply(B, C, BC.data(), N);

    // calculate result matrices
    gpu_matrix_subtract(AC.data(), BD.data(), E, N);
    gpu_matrix_add(AD.data(), BC.data(), F, N);
}

void cpu_complex_matrix_multiply(float *A, float *B, float *C, float *D, float *E, float *F, int N)
{
    // temp matrices
    vector<float> AC(N * N);
    vector<float> BD(N * N);
    vector<float> AD(N * N);
    vector<float> BC(N * N);

    // calculate temp matrices
    cpu_matrix_multiply(A, C, AC.data(), N);
    cpu_matrix_multiply(B, D, BD.data(), N);
    cpu_matrix_multiply(A, D, AD.data(), N);
    cpu_matrix_multiply(B, C, BC.data(), N);    
    
    // calculate result matrices
    cpu_matrix_subtract(AC.data(), BD.data(), E, N);
    cpu_matrix_add(AD.data(), BC.data(), F, N);
}

void gpu_matrix_multiply(float *A, float *B, float *C, int N)
{
    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
    {   
        #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                    C[i * N + j] = 0.0f;
                    for (int k = 0; k < N; k++)
                        C[i * N + j] += A[i * N + k] * B[k * N + j];
                }
    }
}

void cpu_matrix_multiply(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = 0.0f;
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
        }
}

void gpu_matrix_subtract(float *A, float *B, float *C, int N)
{
    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
    {
        #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    C[i * N + j] = A[i * N + j] - B[i * N + j];
    }
}

void cpu_matrix_subtract(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i*N+j] = A[i*N+j] - B[i*N+j];
}

void gpu_matrix_add(float *A, float *B, float *C, int N)
{
    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
    {
        #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

void cpu_matrix_add(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i*N+j] = A[i*N+j] + B[i*N+j];
}

void fill_matrix(float *mat, int N, float b_cpu)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i*N+j] = i+j+b_cpu;
}

void print_matrix(const char *name, float *mat, int N)
{
    cout << "Matrix " << name << ":" << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << mat[i*N+j] << "   ";

        cout << endl;
    }
}