
#include "mullib.cpp"
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

#define DIM1 2000
#define DIM2 2000
float A[DIM1*DIM2];
float B[DIM2*DIM1];
//float A[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
//float B[] = {1,0,1, 0,5,0, 3,0,1, 2,0,0};

int main()
{
    // dynamic
    // float *A = new float[size];
    // static
    // float A[size];

    srand((unsigned)time(0));

    for (int i = 0; i < DIM1*DIM2; i++)
    {
        A[i] = (rand()%100)+1;
        B[i] = (rand()%100)+1;
    }

    if (matrix_multiply_test(A, B, DIM1, DIM2, DIM1) == 0)
    //if (matrix_multiply_test(A, B, 3, 4, 3) == 0)
        cout << "Test PASSED" << endl;
    else
        cout << "Test FAILED" << endl;

    return 0;
}
