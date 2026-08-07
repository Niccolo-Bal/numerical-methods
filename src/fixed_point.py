
# The equation (exp(x) + exp(-x)) / x = exp(x) can be solved as
# x = 1 + exp(-2x), thus g(x) = 1 + exp(-2x)

import numpy as np
from collections.abc import Callable

class FPIterations():
    
    def solve(self, g: Callable[[float], float], error: float = 0.001, x: float = 0, max_n: int = 100, n: int = 0) -> float:
        
        if abs(g(x) - x) < error or n == max_n:
            return x
        
        return self.solve(g, error, g(x), n + 1)

fpi = FPIterations()    

def main():
    g = lambda x: (
        1 + np.exp(-2*x)
    )

    for x_0 in (-1, 1, 10):
        solution = fpi.solve(g, x = x_0)
        print(f"Fixed point solution for x_0 = {x_0}: {solution}")

if __name__ == "__main__":
    main()