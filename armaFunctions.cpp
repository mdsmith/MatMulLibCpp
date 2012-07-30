#include <armadillo>
#include <sys/time.h>
using namespace std;
using namespace arma;

float* arma_matrix_multiply(float A[], float B[], int ah, int ud, int bw)
{
    float *C = new float[ah*bw];

    mat mA = randu<mat>(5,1);
    mA.zeros(ah*ud, 1);
    for (int i = 0; i < ah*ud; i++)
        mA[i] = A[i];
    mA.reshape(ud, ah);
    mA = mA.t();
    //mA.print();


    mat mB = randu<mat>(5,1);
    mB.zeros(ud*bw, 1);
    for (int i = 0; i < ud*bw; i++)
        mB[i] = B[i];
    mB.reshape(bw, ud);
    mB = mB.t();
    //mB.print();


    timeval t1, t2;
    double elapsedTime;
    gettimeofday(&t1, NULL);

    mat mC = mA*mB;

    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec -t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    cout << "Multiplication: " << elapsedTime << " ms.\n";


    //mC.print();
    mC = mC.t();
    mC.reshape(ah*bw, 1);
    for (int i = 0; i < ah*bw; i++)
        C[i] = mC[i];

    return C;
}
