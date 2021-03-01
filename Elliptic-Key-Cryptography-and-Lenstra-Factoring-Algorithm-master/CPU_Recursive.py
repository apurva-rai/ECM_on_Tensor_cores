from numba import vectorize, jit, cuda, guvectorize, void, float64
import numpy as np
import cupy as cp
# to measure exec time
from timeit import default_timer as timer

@jit
def xgcd2(a, b):
    # Base Case
    if a == 0 :
        return b,0,1

    gcd,x1,y1 = xgcd2(b%a, a)

    # Update x and y using results of recursive
    # call
    x = y1 - (b//a) * x1
    y = x1

    return gcd,x,y

@jit
def modinv2(a, b):
    """
    return x such that (x * a) % b == 1
    credit https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    """
    g, x, _ = xgcd2(a, b)
    return x % b, g

@vectorize(['int32(int32, int32)',
            'int64(int64, int64)',
            'float32(float32, float32)',
            'float64(float64, float64)'], target="parallel")
def func2(x,y):
    return modinv2(x,y)[0]

if __name__=="__main__":

    n = 2**22
    a = np.random.randint(1,11,size=n)
    b = np.random.randint(1,11,size=n)

    start = timer()
    func2(a,b)
    print("Time:", timer()-start)
