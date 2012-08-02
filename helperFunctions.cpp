#include "helperFunctions.h"

// definitions
int round_up(int number, int base)
{
    if (number == round_down(number, base))
        return number;
    else
        return round_down(number+base, base);
}

int round_down(int number, int base)
{
    return (int)(number/base)*base;
}

float* pad(float array[], int row_num, int col_num, int base)
{
    int round_col = round_up(col_num, base);
    int round_row = round_up(row_num, base);
    float *tbr = new float[round_col * round_row];
    for (int i = 0; i < round_row; i++)
    {
        for (int j = 0; j < round_col; j++)
        {
            if (i < row_num && j < col_num)
                tbr[i * round_col + j] = array[i * col_num + j];
            else
                tbr[i * round_col + j] = 0.0f;
        }
    }
    return tbr;
}

