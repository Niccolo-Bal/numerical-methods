import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Callable, Optional

from integrals_roots import integrals



class Sampling:

    def __init__(self):
        self.rand = np.random.default_rng()
        self._z_cache = {}
        self._normalized_cache = {}


    # Returns float with normal distribution
    def norm(self, mu: float = 0, std = 1) -> float:
        return self.rand.normal(loc = mu, scale = std)

    # MCMC sample given a dataset, not using the first 20 values to remove some starting bias, 
    # but, especially for functions not centered near x_0, still holds predictable bias for small n
    def mcmc_sample(self, energy: Callable[[float], float], n: int = 10**3, m = 20, thining_factor: int = 1,
                     x_0: float = 0.0,) -> np.ndarray[float]:

        if thining_factor < 1:
            raise ValueError("thining must be a positive integer")

        x = x_0
        count = 0
        samples = np.zeros(n)

        for k in range(n * thining_factor + m):
            proposed = x + self.norm()
            
            prob = np.exp((energy(x) - energy(proposed)))

            if self.rand.random() < prob:
                x = proposed
            if k >= m and k % thining_factor == 0:
                # NOTE: writeup says to add the point only if the new one is accepted
                # but this seems to skew the data and to my understanding is not how MCMC works
                samples[count] = x
                count += 1

        return samples


    # Private func to normalize energy function into distribution
    def normalized(self, energy: Callable[[float], float], bounds: 
                    tuple[float, float] = (-10, 10)) -> Callable[[float], float]:
        if energy not in self._z_cache:
            self._z_cache[energy] = integrals.midpoint_area(
                lambda x: np.exp(-energy(x)), bounds[0], bounds[1], 20_000)
        z = self._z_cache[energy]
        return lambda x: np.exp(-energy(x)) / z


    # CDF using midpoint
    def cdf(self, x: float, distribution: Optional[Callable[[float], float]] = None, 
            energy: Optional[Callable[[float], float]] = None) -> float:

        if distribution is None:
            if energy in self._normalized_cache:
                distribution = self._normalized_cache[energy]
            elif energy is None:
                raise ValueError("cdf requires distrubution or energy function paramater")
            else: 
                distribution = self.normalized(energy)
                self._normalized_cache[energy] = distribution

        return integrals.midpoint_area(distribution, -10, x, 10_000)


    def inverse_cdf(self, k: float, distribution: Optional[Callable[[float], float]] = None, 
            energy: Optional[Callable[[float], float]] = None) -> float:

        cdf_offset = lambda x: self.cdf(x, distribution) - k

        return integrals.midpoint_root(cdf_offset)


    def cdf_sampling(self, distribution: Optional[Callable[[float], float]] = None, 
            energy: Optional[Callable[[float], float]] = None, n: int = 10**3) -> np.ndarray[float]:
        
        samples = np.zeros(n)
        for i in range(n):
            samples[i] = self.inverse_cdf(self.rand.random(), distribution, energy)
        return samples




samp = Sampling()


def main():
    #### Testcase variables

    v = lambda x: (
        (x**2 / 10) + (np.sin(x)) + (np.exp(-5*(x**2)))
    )   

    z = integrals.midpoint_area(lambda x: np.exp(-v(x)), -10, 10, 10_000)

    # Normalized v
    p = lambda x: (
        (1 / z) * (np.exp(-v(x)))
    )

    ####

    # Anneling example + benchmark
    anneling_start = time.perf_counter_ns()
    print("Annealing min (x, V(x), k): ", samp.annealing(v))
    print("Time for annealing: "
          f"{(time.perf_counter_ns() - anneling_start)*1e-6:.2g} ms")

    print()

    # Sampling example + benchmark
    n = 1_000
    xlim = (-8, 8)
    for sampling_type in ("MCMC", "Inverse CDF"):

        sampling_start = time.perf_counter_ns()

        samples = (
            samp.mcmc_sample(v, n = n, thining_factor = 3) if sampling_type == "MCMC"
            else samp.cdf_sampling(distribution = p, n = n)
        )
        print(f"Sampling time for {sampling_type}: "
              f"{(time.perf_counter_ns() - sampling_start)*1e-6:.4g} ms")

        plt.hist(samples, bins = 51, density = True, edgecolor = "skyblue", range = xlim, label = f"{sampling_type} Sample")
        plt.plot(np.linspace(-7, 7, 1000), [p(x) for x in np.linspace(-7, 7, 1000)], label = "$p(x)$")
        plt.xlim(xlim)
        plt.legend()
        plt.show()

        print()


if __name__ == "__main__":
    main()

