#include "naiveFunctions.h"
#include <iostream>
using namespace std;

float* naive_matrix_multiply(float A[], float B[], int ah, int ud, int bw)
{
    float *C = new float[ah * bw];
    for (int i = 0; i < ah; i++)
    {
        for (int j = 0; j < bw; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < ud; l++)
            {
                sum += A[i*ud + l] * B[l*bw + j];
            }
            C[i*bw+j] = sum;
        }
    }
    return C;
}

float* omp_matrix_multiply(float A[], float B[], int ah, int ud, int bw)
{
    float *C = new float[ah * bw];
    float sum = 0.0f;
    #pragma omp parallel for private(sum)
    for (int i = 0; i < ah; i++)
    {
        for (int j = 0; j < bw; j++)
        {
            sum = 0.0f;
            for (int l = 0; l < ud; l++)
            {
                sum += A[i*ud + l] * B[l*bw + j];
            }
            C[i*bw+j] = sum;
        }
    }
    return C;
}
