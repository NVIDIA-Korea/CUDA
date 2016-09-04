
int main(int argc, char **argv)
{
  float *x, *y, tmp;
  int n = 1<<20, i;

  x = (float*)malloc(n*sizeof(float));
  y = (float*)malloc(n*sizeof(float));

  #pragma acc data create(x[0:n]) copyout(y[0:n])
  {
    #pragma acc kernels
    {
      for( i = 0; i < n; i++)
      {
        x[i] = 1.0f;
        y[i] = 0.0f;
      }
    }

    #pragma acc host_data use_device(x,y)
    {
      cublasSaxpy(n, 2.0, x, 1, y, 1);
    }
  }

  fprintf(stdout, "y[0] = %f\n",y[0]);
  return 0;
}