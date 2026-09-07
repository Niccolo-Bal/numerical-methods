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
    def descent_1D(self, loss: Callable[[float], float], x_0: float = 0.0,h: float = 1e-6, alpha: float = 1e-3,
                    max_iter: int = 10**4, error: float | None = None) -> tuple[float, float, int]:

        x = x_0
        cur_loss = loss(x)

        for _ in range(max_iter):
            gradient = (loss(x + h) - cur_loss) / h
            x -= alpha * gradient
            new_loss = loss(x)
            if error and abs(cur_loss - new_loss) <= error:
                break
            cur_loss = new_loss

        return (x, cur_loss)

    def descent_2D(self, loss: Callable[[float, float], float], x_0: float = 0.0, y_0: float = 0.0, h: float = 1e-6, 
                   alpha: float = 1e-3, max_iter: int = 20**4, error: float | None = None,) -> tuple[tuple[float, float], float, int]:

        x, y = x_0, y_0
        cur_loss = loss(x, y)

        for _ in range(max_iter):
            x_partial = (loss(x + h, y) - cur_loss) / h
            y_partial = (loss(x, y + h) - cur_loss) / h
            x -= alpha * x_partial
            y -= alpha * y_partial
            new_loss = loss(x, y)
            if error and abs(cur_loss - new_loss) <= error:
                break
            cur_loss = new_loss

        return (x, y, cur_loss)


    # Find approx minimum for a function using simulated anneling
    def annealing(self, loss: Callable[[float], float], x_0: float = 0.0, max_iter: int = 10**3,
                  alpha: float = 0.5, temp: Callable | None = None) -> tuple[float, float, int]:

        if temp is None:
            temp = self.temp

        x = x_0
        x_loss = loss(x)

        mn_pair = (x_0, x_loss, 0)

        for k in range(max_iter):
            proposed = x + (samp.norm() * alpha)

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

    # Gradient example + benchmark
    print(mini.descent_1D(v, 1))

    # Combined
    print(mini.descent_1D(v, anneling_out[0]))


if __name__ == "__main__":
    main()