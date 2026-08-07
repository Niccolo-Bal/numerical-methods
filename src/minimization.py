import numpy as np
import time
from typing import Callable
from sampling import samp

class Minimization:

    def __init__(self, temp: Callable[[float], float] | None = None):

        if temp is None:
            # Temp where k is the kth - 1 iteration (0-indexed)
            temp = lambda k: (10 / (np.log(k + 2)))

        self.temp = temp

    # Basic gradient descent for 1 dimensional function
    def descent_1d(self, loss: Callable[[float], float], x_0: float = 0.0,h: float = 1e-6, step: float = 1e-2,
                    iterations: int = 10**3) -> tuple[float, float, int]:
    
            x = x_0
            x_loss = loss(x_0)
    
            for _ in range(iterations):
                gradient = (loss(x + h) - x_loss) / h
                x -= step * gradient
                x_loss = loss(x)
    
            return (x, x_loss)

    def descent_2d(self, loss: Callable[[float, float], float], x_0: float = 0.0,h: float = 1e-6, step: float = 1e-2,
                    iterations: int = 10**3) -> tuple[tuple[float, float], float, int]:
        pass

    # Find approx minimum for a function using simulated anneling
    def annealing(self, loss: Callable[[float], float], x_0: float = 0.0, iterations: int = 10**3,
                  temp: Callable | None = None) -> tuple[float, float, int]:

        if temp is None:
            temp = self.temp

        x = x_0
        x_loss = loss(x)

        mn_pair = (x_0, x_loss, 0)

        for k in range(iterations):
            proposed = x + samp.norm()

            proposed_loss = loss(proposed)

            prob = np.exp((x_loss - proposed_loss)
                            / (temp(k)))

            if samp.rand.random() < prob:
                if proposed_loss < mn_pair[1]:
                    mn_pair = (proposed, proposed_loss, k + 1)
                x = proposed
                x_loss = proposed_loss

        return mn_pair



mini = Minimization()

def main():
    v = lambda x: (
        (x**2 / 10) + (np.sin(x)) + (np.exp(-5*(x**2)))
    )

    # Anneling example + benchmark
    anneling_start = time.perf_counter_ns()
    anneling_out = mini.annealing(v)
    print("Annealing min (x, V(x), k): ", anneling_out)
    print("Time for annealing: "
          f"{(time.perf_counter_ns() - anneling_start)*1e-6:.2g} ms")
    print(mini.descent_1d(v, 0))
    print(mini.descent_1d(v, anneling_out[0]))


if __name__ == "__main__":
    main()