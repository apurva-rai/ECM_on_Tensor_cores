from numba import vectorize, jit, cuda, guvectorize, void, float64
import numpy as np
import cupy as cp
# to measure exec time
from timeit import default_timer as timer

@jit
def xgcd2(a, b):
    """
    return (g, x, y) such that a*x + b*y = g = gcd(a, b)
    credit https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    """
    x0, x1, y0, y1 = 0, 1, 1, 0
    while a != 0:
        (q, a), b = divmod(b, a), a
        y0, y1 = y1, y0 - q * y1
        x0, x1 = x1, x0 - q * x1
    return b, x0, y0

@jit
def modinv2(a, b):
    """
    return x such that (x * a) % b == 1
    credit https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    """
    g, x, _ = xgcd2(a, b)
    return x % b, g

def func1(x,y):
    for i in range(2**29):
        modinv(x[i],y[i])

@vectorize(['int32(int32, int32)',
            'int64(int64, int64)',
            'float32(float32, float32)',
            'float64(float64, float64)'], target="parallel")
def func2(x,y):
    return modinv2(x,y)[0]

if __name__=="__main__":

    n = 2**29
    a = np.random.randint(1,11,size=n)
    b = np.random.randint(1,11,size=n)

    start = timer()
    func2(a,b)
    print("Time:", timer()-start)
