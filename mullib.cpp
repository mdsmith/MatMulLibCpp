
#include "naiveFunctions.cpp"
#include "oclFunctions.cpp"
#include <iostream>
using namespace std;

int matrix_multiply_test(float A[], float B[], int ah, int ud, int bw)
{
    float *goldenC = naive_matrix_multiply(A, B, ah, ud, bw);
    for (int i = 0; i < ah; i++)
    {
        for (int j = 0; j < bw; j++)
            printf("%4.0f", goldenC[i * bw + j]);
        cout << endl;
    }

    return 0;
}


