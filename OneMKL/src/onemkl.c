#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "mkl_types.h"
#include "mkl_cblas.h"
#include <time.h>
#include <omp.h>

#define ROUND 9
#define MEDIAN 5

int main(int argc, char *argv[]) {

    if (argc != 5) {
        printf("Input Error, usage: ./onemkl <THREAD_NUM> M N K\n");
        return 1;
    }

    int thread_num = atoi(argv[1]);
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int k = atoi(argv[4]);

    srand(149);
    if (thread_num) {
        mkl_set_num_threads(thread_num);
        printf("Setting thread_num=%d\n", thread_num);
    }
    
    double start, end, total_time = 0, mean_time, median_time, time_array[ROUND];
    uint32_t sizeofA = m * k;
    uint32_t sizeofB = k * n;
    uint32_t sizeofC = m * n;

    float *A = (float*)malloc(sizeof(float) * sizeofA);
    float *B = (float*)malloc(sizeof(float) * sizeofB);
    float *C = (float*)malloc(sizeof(float) * sizeofC);

    for (int i = 0; i < sizeofA; i++)   A[i] = ((float)rand()) / RAND_MAX;
    for (int i = 0; i < sizeofB; i++)   B[i] = ((float)rand()) / RAND_MAX;
    for (int i = 0; i < sizeofC; i++)   C[i] = ((float)rand()) / RAND_MAX;

    int lda = k, ldb = n, ldc = n;
    for (int round = 0; round < ROUND; round++) {
        start = omp_get_wtime();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, lda, B, ldb, 1, C, ldc);
        end = omp_get_wtime();
        time_array[round] = end - start;
        total_time += time_array[round];
    }
    mean_time = total_time / ROUND;
    
    // get the median_time
    int min_index;
    double temp_time, min_time;
    for (int i = 0; i < MEDIAN; i++) {
        min_time = time_array[i];
        min_index = i;
        
        for (int j = i+1; j < ROUND; j++) {
            if (time_array[j] < min_time) {
                min_time = time_array[j];
                min_index = j;
            }
        }
        if (min_index != i) {
            temp_time = time_array[i];
            time_array[i] = min_time;
            time_array[min_index] = temp_time;
        }
    }
    median_time = time_array[MEDIAN - 1];

    double performance_median = 2.0e-9 * m * n * k / median_time;
    double performance_mean = 2.0e-9 * m * n * k / mean_time;
    printf("M=%d N=%d K=%d\tTHREAD_NUM=%d performance_mean = %.0f GFLOPs\n", m, n, k, thread_num, performance_mean);
    printf("M=%d N=%d K=%d\tTHREAD_NUM=%d performance_median = %.0f GFLOPs\n", m, n, k, thread_num, performance_median);
    printf("\n");
    free(A);
    free(B);
    free(C);
    return 0;

}
