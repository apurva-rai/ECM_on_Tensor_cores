import math
import random
from timeit import default_timer as timer
import concurrent.futures

def xgcd(a, b):
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

def modinv(a, b):
    """
    return x such that (x * a) % b == 1
    credit https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    """
    g, x, _ = xgcd(a, b)
    return x % b, g

def func2(a,b):
    for i in range(len(a)):
        modinv(a[i],b[i])

if __name__=="__main__":

    k = 2**22

    c = [random.randint(1,11) for _ in range(k)]
    d = [random.randint(1,11) for _ in range(k)]
    start = timer()
    func2(c,d)
    print("Time:", timer()-start)
