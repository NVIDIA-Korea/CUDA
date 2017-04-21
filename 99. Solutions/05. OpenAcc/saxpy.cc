
#include <stdio.h>
#include <stdlib.h>

void saxpy_parallel(int n, float a, float *x, float *y)
{
    #pragma acc kernels
    for (int i = 0; i < n; ++i)
        y[i] = a*x[i] + y[i];
}    

int main(int argc, char **argv)
{
  float *x, *y, tmp;
  int n = 1<<6, i;

  x = (float*)malloc(n*sizeof(float));
  y = (float*)malloc(n*sizeof(float));

  #pragma acc data create(x[0:n]) copyout(y[0:n])
  {
    #pragma acc kernels
    {
      for( i = 0; i < n; i++)
      {
        x[i] = 0.5f * i;
        y[i] = 0.2f * i;
      }
    }

    saxpy_parallel(n, 2.0, x, y);
  }

    for (i = 0; i < n; ++i) {
      printf("%f ", y[i]);
    }
  return 0;
}