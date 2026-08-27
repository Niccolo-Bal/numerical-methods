
import numpy as np
from collections.abc import Callable
from pseudorandom_numbers import LCG

lcg = LCG(0.1)

class Integrals():

    midpoint_iterations = 0
    newton_iterations = 0

    # Approximates intergal of f over [lower_bound, upper_bound] using midpoint formula over n evenly-spaced points
    def midpoint_area(self, f: Callable[[float], float], lower_bound: float, upper_bound: float, n: int = 1000) -> float:
        x = np.linspace(lower_bound, upper_bound, n) if n > 1 else lower_bound
        y = f(x)

        return np.average(y) * (upper_bound - lower_bound)
    
    # Approximate integral of f using n randomly sampled points over [lower_bound, upper_bound] 
    def monte_carlo(self, f: Callable[[float], float], lower_bound: float, upper_bound: float, n: int = 10_000, 
                    lower_y: float | None = None, upper_y: float | None = None) -> float:
        
        if lower_y is None or upper_y is None:
            x = np.linspace(lower_bound, upper_bound, 1000)
            y = f(x)
            if lower_y is None:
                lower_y = np.min(y) - 0.1 * (np.max(y) - np.min(y))
            if upper_y is None:
                upper_y = np.max(y) + 0.1 * (np.max(y) - np.min(y))

        samples = np.zeros(n, dtype = int)
        area = (upper_y - lower_y) * (upper_bound - lower_bound)

        for i in range(n):
            x_i = lower_bound + lcg.lcg() * (upper_bound - lower_bound)
            lcg.lcg()
            y_i = lower_y + lcg.lcg() * (upper_y - lower_y)

            if y_i >= 0:
                if y_i <= f(x_i):
                    samples[i] = 1
            else:
                if y_i >= f(x_i):
                    samples[i] = -1
            
        return np.average(samples) * area
    
    # Uses midpoint guess-and-check to find root inbetween mixed-sign outputs
    # a and b (f(a) and f(b) hold different sizes) recursively
    def midpoint_root(self, f: Callable[[float], float], error: float = 10**(-3), 
                      a: float = (lcg.lcg() - 0.5) * 10, b: float | None = None) -> float:
        
        self.midpoint_iterations += 1

        if b is None:
            scale = 2
            while True:
                b = (lcg.lcg()) * scale
                scale += scale
                if f(b) * f(a) < 0:
                    break
                if f(-b) * f(a) < 0:
                    b = -b
                    break
                if scale >= 2**16:
                    raise ValueError(f"No {"positive" if f(b) < 0 else "negative"} values found"
                                     " on (-2^16, 2^16), use Newtons for subtler functions")
                
        if f(b) * f(a) >= 0:
            raise ValueError("f(a) and f(c) must have opposite signs")
        
        mid = min(b, a) + 0.5 * abs(b - a)

        if abs(f(mid)) < error:
            return mid
        
        if f(mid) * f(a) < 0:
            return self.midpoint_root(f, error, a, mid)
        
        else:
            return self.midpoint_root(f, error, mid, b)
        
    # basic central difference
    def derivative(self, f: Callable[[float], float], x: float, h: float = 10**(-3)) -> float:
        return (f(x + h) - f(x - h)) / (2 * h)

    # Uses Newtons method to find a root of a function
    def newton_root(self, f: Callable[[float], float], x_n: float = (lcg.lcg() - 0.5) * 10, error: float = 10**(-3)) -> float:

        self.newton_iterations += 1

        x_n = x_n - (f(x_n) / self.derivative(f, x_n))

        if abs(f(x_n)) < error:
            return x_n

        return self.newton_root(f, x_n, error)

integrals = Integrals()


def main() -> None:
    g = lambda x: (
        (np.exp(-x**2 / 2)) / 
        (np.sqrt(2*np.pi))
    )

    f_1 = lambda z: (
        integrals.midpoint_area(g, -5, z, 200)
    )

    f_2 = lambda z: (
        integrals.monte_carlo(g, -5, 5, 10_000)
    )

    f_minus = lambda z: (
        f_1(z) - 0.5
    )

    print(integrals.midpoint_root(f_minus))
    print(integrals.newton_root(f_minus))

    print(integrals.midpoint_iterations)
    print(integrals.newton_iterations)


if __name__ == "__main__":
    main()