#include <iostream>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
//#include <mpir.h>

using namespace std;

struct Point
{

int x = 0;
int y = 0;

};

__device__
int xgcd(int a, int b, int *x, int *y)
{

  int prevx = 1, x1 = 0, prevy = 0, y1 = 1;

  while (b)
  {

    int q = a/b;

    prevx = x1;
    x1 = prevx - q*x1;
    prevy = y1;
    y1 = prevy - q*y1;
    b = a % b;
    a = b;

  }

  *x = prevx;
  *y = prevy;

  return a;

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

    if (g != 1) ;
    else ((a%y[i] + y[i]) % y[i]);

  }
}
/*
Point elliptic_add(Point P, Point Q, int a, int b, int n)
{

    if (P.x == 0 && P.y == 0) return Q;
    else if(Q.x == 0 && Q.y == 0) return P;

    int x1 = P.x;
    int y1 = P.y;
    int x2 = Q.x;
    int y2 = Q.y;
    int num = 0, den = 0;

    if(P.x == Q.x && P.y == Q.y)
    {

      if (2*y1%n == 0)
      {

        Point temp;
        return temp;

      }
      else
      {

        num = (3*(pow(x1,2) + a) % n;
        den = (2*y1) % n;

      }

    }
    else
    {

      if((x2-x1)%n == 0)
      {

        Point temp;
        return temp;

      }
      else
      {

        num = (y2- y1) % n
        den = (x2- x1) % n

      }

    }

    if(__gcd(den, n) != 1)
    {

      Point temp;
      temp.x = -1;
      temp.y = -1;

      return temp;

    }
    else den = modinv(den,n);

    int slope = (num * den) % n;
    int xR = (pow(slope,2)%n) -x1 -x2) %n;
    int yR = (slope*(x1- ((pow(slope,2)%n) - x1 - x2)) - y1) % n

    Point temp2;
    temp2.x = xR;
    temp2.y = yR;

    return temp2;

}

Point elliptic_mul(int d, Point P, int a, int n)
{
  int b=0;

  if(P.x != 0 && P.y != 0) b = (pow(P.y,2) - pow(P.x,3) - a*P.x) % n;

  Point tempR;
  tempR.x = P.x;
  tempR.y = P.y;

  while(d > 1)
  {

    Point temp = elliptic_add(P,tempR,a,b,n);
    tempR.x = temp.x;
    tempR.y = temp.y;

    d-= 1;

  }

  return tempR;

}

int lenstra(Point P, int a, int n)
{

  int i = 0;

  while(i < n)
  {

    Point tempR;
    tempR = elliptic_mul(int(tgamma(i+1)), P, a, n);
    i += 1;

    if(tempR.x == -1 && tempR.y == -1) return n;

    if(__gcd((2*tempR.y)%n,n) != 1) return (__gcd(2*tempR.y,n)%n);

  }

  return n;

}

Point rand_elliptic(int n, int &a)
{

  bool state = 0;

  while(state != 1)
  {

    Point P0;
    srand((unsigned)time(NULL));
    P0.x = rand()%n + 1;
    P0.y = rand()%n + 1;
    a = rand()%n + 1;

    int b = (pow(P0.y,2) - pow(P0.x,3) - a*P0.x) % n;
    state = ((pow(P0.y,2))%n == (pow(P0.x,3))%n + ((a*P0.x)%n) + b);

  }

  return P0;

}

int lenstra_random(int n)
{

  int g = n;
  int a = 0;

  while(g == n)
  {

    Point P0 = rand_elliptic(n,a);
    g = lenstra(P0,a,n);
    int b = (pow(P0.y,2) - pow(P0.x,3) - a*P0.x) % n;

  }

  return g;

}
*/
int main(void)
{
  int N = 1<<27;
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
