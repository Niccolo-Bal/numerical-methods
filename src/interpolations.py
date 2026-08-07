
import numpy as np
import functools
import matplotlib.pyplot as plt
from collections.abc import Callable

class Interpolations:

    # given f, generate N evenly spaced points on interval
    @functools.lru_cache() 
    def generate_points(self, f: Callable[[float], float], n: int, interval: tuple[float, float]) -> list[tuple[float, float]]:
        
        points = []

        for x_i in np.linspace(interval[0], interval[1], n):
            points.append((x_i, f(x_i)))

        return points
    
    # given f, generate N points with a chebyshev distribution (cosine)
    def generate_cheby(self, f: Callable[[float], float], n: int, interval: tuple[float, float]) -> list[tuple[float, float]]:
        
        points = []
        pi = np.pi
        scale = interval[1] - interval[0]

        for i in range(1, n + 1):
            t_i = interval[0] + (np.cos((2*i - 1) * pi / (2 * n)) + 1) * 0.5 * scale
            points.append((t_i, f(t_i)))

        return points

    # returns a list of N polynomial coefficients in desending order for a function that passes through N evenly 
    # spaced points on f(x) over interval (x, y). O(n^2)
    def lagrange(self, f: Callable[[float], float], n: int, interval: tuple[float, float], chevy: bool = False) -> list[float]:
        
        coef = np.array([0.0] * n)
        points = self.generate_points(f, n, interval) if not chevy else self.generate_cheby(f, n, interval)

        for x_i, y_i in points:
            denom = 1

            for x_j, _ in points:
                if x_j == x_i:
                    continue
                denom *= (x_i - x_j)

            partials = [y_i / denom]

            for root, _ in points:
                if root == x_i:
                    continue
                partials = ( # Multiply (x - x_j) against existing partial equation
                    [partials[0]] + [partials[i+1] - root*partials[i] for i in range(len(partials)-1)]
                    + [-root*partials[-1]]
                )
            
            coef += np.array(partials)
        
        return list(coef)
    
    # Compares (plots) a function against its polynomial and piecewise-linear approximations 
    def compare_interpolations(self, f: Callable[[float], float], interval: tuple[float, float], n: int,
                               plot: bool = True, true_function: bool = True, polynomial: bool = False, 
                               cheby: bool = False, piecewise: bool = False) -> dict[str, tuple[float, float]]:
        

        x = np.linspace(interval[0], interval[1], 1000)
        real_y = np.array([f(x_i) for x_i in x])
        errors = {}

        if true_function:
            plt.plot(x, real_y, label = "f(x)")

        if polynomial: 
            interp = np.polynomial.Polynomial(list(reversed(self.lagrange(f, n, interval))))
            poly_y = interp(x)
            poly_error = np.abs(poly_y - real_y)
            errors["polynomial"] = (np.max(poly_error), np.average(poly_error))
            plt.plot(x, poly_y, label = "uniform polynomial approximation", linestyle = ":", color = "red")

        if cheby:
            interp = np.polynomial.Polynomial(list(reversed(self.lagrange(f, n, interval, True))))
            cheby_y = interp(x)
            cheby_error = np.abs(cheby_y - real_y)
            errors["chebyshev"] = (np.max(cheby_error), np.average(cheby_error))
            plt.plot(x, cheby_y, label = "Chebyushev polynomial approximation", linestyle = ":", color = "green")

        if piecewise:
            points = self.generate_points(f, n, interval)
            piecewise_y = np.interp(x, np.array([points[i][0] for i in range(n)]), np.array([points[i][1] for i in range(n)]))
            piecewise_error = np.abs(piecewise_y - real_y)
            errors["piecewise"] = (np.max(piecewise_error), np.average(piecewise_error))
            plt.plot(x, piecewise_y, label = "Piecewise interpolation", linestyle = "dashed", color = "teal")

        plt.legend()

        if plot:
            plt.plot()

        return errors

interp = Interpolations()

def main():
    f = lambda x: (
        1 / 
        (np.exp(-x) + np.exp(x)) 
    )
    interval = (-5, 5)

    errors = interp.compare_interpolations(f, interval, 15)


    print(f"Uniform distribution max error: {errors["polynomial"][0]}. Average: {errors['polynomial'][1]}" if "polynomial" in errors else "", 
          f"Cheby distribution max error: {errors["chebyshev"][0]}. Average: {errors['chebyshev'][1]}" if "chebyshev" in errors else "",
          f"Piecewise distribution max error: {errors["piecewise"][0]}. Average: {errors['piecewise'][1]}" if "piecewise" in errors else "", sep = "\n")

    plt.show()

    
if __name__ == "__main__":
    main()