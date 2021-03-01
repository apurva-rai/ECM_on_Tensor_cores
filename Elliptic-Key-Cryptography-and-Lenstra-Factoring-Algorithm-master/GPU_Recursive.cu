#include <iostream>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

using namespace std;

__device__
int xgcd(int a, int b, int *x, int *y)
{

  // Base Case
   if (a == 0)
   {
       *x = 0, *y = 1;
       return b;
   }

   int x1, y1; // To store results of recursive call
   int gcd = xgcd(b%a, a, &x1, &y1);

   // Update x and y using results of recursive
   // call
   *x = y1 - (b/a) * x1;
   *y = x1;

   return gcd;

}

__global__
void modinv(int *x, int *y, int n)
{

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {

    int a, b;
    int g = xgcd(x[i], y[i], &a, &b);

    if (g != 1) return;
    else (a%y[i] + y[i]) % y[i];

  }
}

int main(void)
{
  int N = 1<<22;
  int *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(int));
  cudaMallocManaged(&y, N*sizeof(int));

  srand((unsigned)time(NULL));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = rand()%10+1;
    y[i] = 11;

  }

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  modinv<<<numBlocks, blockSize>>>(x, y, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
