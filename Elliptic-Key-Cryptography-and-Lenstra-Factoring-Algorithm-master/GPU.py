from sympy.ntheory import factorint
# to measure exec time
from timeit import default_timer as timer

start = timer()
n = factorint(134165873)
print(n, timer()-start)
