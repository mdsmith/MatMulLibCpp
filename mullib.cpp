
#include "naiveFunctions.cpp"
#include "oclFunctions.cpp"
#include <iostream>
#include <sys/time.h>
using namespace std;

int matrix_multiply_test(float A[], float B[], int ah, int ud, int bw)
{

    timeval t1, t2;
    double elapsedTime;

    gettimeofday(&t1, NULL);

    float *goldenC = naive_matrix_multiply(A, B, ah, ud, bw);
    /*
    for (int i = 0; i < ah; i++)
    {
        for (int j = 0; j < bw; j++)
            printf("%4.0f", goldenC[i * bw + j]);
        cout << endl;
    }

    */
    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec -t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    cout << elapsedTime << " ms.\n";

    gettimeofday(&t1, NULL);

    float *C = ocl_matrix_multiply(A, B, ah, ud, bw);
    /*
    for (int i = 0; i < ah; i++)
    {
        for (int j = 0; j < bw; j++)
            printf("%4.0f", C[i * bw + j]);
        cout << endl;
    }
    */
    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec -t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    cout << elapsedTime << " ms.\n";

    gettimeofday(&t1, NULL);

    return 0;
}


