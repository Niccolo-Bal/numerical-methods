import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time, math
from numpy import random, linalg

class Matricies:


    def random_b(self, n: int = 3, interval: tuple[float] = (-5, 5)) -> np.ndarray[float]:
        return np.array([random.random() * (interval[1] - interval[0]) + interval[0] for _ in range(n)])


    def construct_random_sparse(self, n: int = 3) -> np.ndarray[np.ndarray]:
        mtrx = np.array([np.zeros(n) for _ in range(n)])
        prob = 1 / (n**(2/3))

        for i in range(n):
            sm = 0
            for j in range(n):
                if i == j:
                    continue

                if random.random() < prob:
                    val = random.random() * 2 - 1
                    mtrx[i][j] = val
                    sm += abs(val)
            mtrx[i][i] = sm + 1

        return mtrx


    # Computes A @ x with an explicit loop rather than the @ operator. We do this because we want to
    # compare Python vs Python; the COO product below has no real equivalent in Numpy's API and so
    # must be written by hand, and pitting Python vs BLAS would not measure the high-level algorithm, but just
    # other optimizations in Numpy.
    def dense_matvec(self, mtrx: np.ndarray[np.ndarray], x: np.ndarray[float]) -> np.ndarray[float]:
        n = len(mtrx)
        av = np.zeros(n)

        for i in range(n):
            sm = 0
            for j in range(n):
                sm += mtrx[i][j] * x[j]
            av[i] = sm

        return av


    # Gets coords and values of all nonzero entries. Stored in an ndarray such that coords[0] is an array of i's, coords[1] 
    # is j's and vals is the array of values. I.e., vals[k] = mtrx[ coords[0][k] ][ coords[1][k] ], where vals[k] is the 
    # kth nonzero entry in the matrix (counting rowwise)
    def get_coo(self, mtrx: np.ndarray[np.ndarray]) -> tuple[np.ndarray[np.ndarray[int]], np.ndarray[float]]:
        n = len(mtrx)
        rows = []
        cols = []
        vals = []

        for i in range(n):
            for j in range(n):
                if mtrx[i][j] != 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(mtrx[i][j])

        return np.array([np.array(rows, int), np.array(cols, int)]), np.array(vals, float)

    # Computes A @ x from the COO triplets.
    def coo_matvec(self, coords: np.ndarray[np.ndarray[int]], vals: np.ndarray[float],
                   x: np.ndarray[float], n: int) -> np.ndarray[float]:
        av = np.zeros(n)
        k = 0

        for i in range(n):
            while k < len(vals) and coords[0][k] == i:
                av[i] += vals[k] * x[coords[1][k]]
                k += 1

        return av


    # Appends one iteration's tracked quantity and the accumulated runtime
    def record(self, history: dict[str, list[float]] | None, key: str, value: float, elapsed: int) -> None:
        if history is None:
            return

        history[key].append(value)
        history["times"].append((history["times"][-1] if history["times"] else 0) + elapsed / 10**6)


    # Assumes n x n matrix
    def jacobi_dense(self, mtrx: np.ndarray[np.ndarray], b: np.ndarray, x: np.ndarray | None = None,
                     error: float = 10**(-4), history: dict[str, list[float]] | None = None) -> np.ndarray[float]:
        n = len(mtrx)
        diag = mtrx.diagonal()

        if x is None:
            x = np.zeros(n)

        start = time.perf_counter_ns()

        next = (b - (self.dense_matvec(mtrx, x) - diag * x)) / diag

        residual = linalg.norm(self.dense_matvec(mtrx, next) - b)
        converged = linalg.norm(next - x) < error or residual < error

        self.record(history, "errors", residual, time.perf_counter_ns() - start)

        if converged:
            return next

        return self.jacobi_dense(mtrx, b, next, error, history)


    def jacobi_coo(self, mtrx: np.ndarray[np.ndarray], b: np.ndarray, x: np.ndarray | None = None,
                     error: float = 10**(-4), history: dict[str, list[float]] | None = None) -> np.ndarray[float]:
        n = len(mtrx)
        cords, vals = self.get_coo(mtrx)
        diag = mtrx.diagonal()

        if x is None:
            x = np.zeros(n)

        def next_x(x: np.ndarray = x) -> np.ndarray:
            start = time.perf_counter_ns()

            next = (b - (self.coo_matvec(cords, vals, x, n) - diag * x)) / diag

            residual = linalg.norm(self.coo_matvec(cords, vals, next, n) - b)
            converged = linalg.norm(next - x) < error or residual < error

            self.record(history, "errors", residual, time.perf_counter_ns() - start)

            if converged:
                return next

            return next_x(next)

        return next_x()


    # Runs both implementations against the same system and returns their per-iteration residual
    # and the total runtime accumulated through that iteration.
    def get_iteration_data_jacobian(self, n: int) -> dict[str, list[float]]:
        mtrx = self.construct_random_sparse(n)
        b = np.ones(n)

        dense = {"errors": [], "times": []}
        coo = {"errors": [], "times": []}

        self.jacobi_dense(mtrx, b, history=dense)
        self.jacobi_coo(mtrx, b, history=coo)

        return { "dense_times" : dense["times"], "coo_times" : coo["times"],
                 "dense_errors" : dense["errors"], "coo_errors" : coo["errors"]}


    # Part 4: norm error and accumulated runtime per iteration, dense against COO
    def compare_jacobian(self, n: int, *, error: bool = True, runtime: bool = True) -> None:
        data = self.get_iteration_data_jacobian(n)

        dense_k = np.arange(1, len(data["dense_errors"]) + 1)
        coo_k = np.arange(1, len(data["coo_errors"]) + 1)

        if error:
            fig, ax = plt.subplots(figsize=(7, 4))

            # Both implementations perform identical arithmetic and so produce identical iterates;
            # the curves coincide exactly, hence the dashed overlay to keep both visible
            ax.plot(dense_k, data["dense_errors"], label="dense", marker="o", markersize=4)
            ax.plot(coo_k, data["coo_errors"], label="COO", marker="x", markersize=6, linestyle="--")

            ax.set_yscale('log')
            ax.legend()
            ax.set_xlabel("Iteration k")
            ax.set_ylabel(r"$\|Ax^{(k)} - b\|_2$")

            plt.tight_layout()
            plt.show()

        if runtime:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(dense_k, data["dense_times"], label="dense", marker="o", markersize=4)
            ax.plot(coo_k, data["coo_times"], label="COO", marker="x", markersize=6, linestyle="--")

            # Log scale, otherwise the COO curve is pinned to the axis and unreadable next to dense
            ax.set_yscale('log')
            ax.legend()
            ax.set_xlabel("Iteration k")
            ax.set_ylabel("Accumulated runtime (ms)")

            plt.tight_layout()
            plt.show()


    def get_scaling_data_jacobian(self, testcases: tuple[int], runs: int) -> dict[str, list[float]]:
        data = { "dense_times" : [], "coo_times" : [], "dense_errors" : [], "coo_errors" : []}

        for n in testcases:
            coo_times, dense_times, coo_errors, dense_errors = [], [], [], []
            
            for _ in range(runs):
                mtrx = self.construct_random_sparse(n)
                b = np.ones(n)
    
                dense_start = time.perf_counter_ns()
                x = self.jacobi_dense(mtrx, b)
                dense_times.append(time.perf_counter_ns() - dense_start)
                dense_errors.append(linalg.norm(mtrx @ x - b))
    
                coo_start = time.perf_counter_ns()
                x = self.jacobi_coo(mtrx, b)
                coo_times.append(time.perf_counter_ns() - coo_start)
                coo_errors.append(linalg.norm(mtrx @ x - b))
    
            data["dense_times"].append((sum(dense_times) / runs) / 10**6); data["coo_times"].append((sum(coo_times) / runs) / 10**6)
            data["dense_errors"].append((sum(dense_errors) / runs) * 10**6); data["coo_errors"].append((sum(coo_errors) / runs) * 10**6)

        return data


    # Supplementary: how the dense/COO gap widens with n. Not required by the assignment
    def scaling_jacobian(self, testcases: tuple[int], runs: int = 10, *,
                         error: bool = True, runtime: bool = True) -> None:
        data = self.get_scaling_data_jacobian(testcases, runs)

        x = np.arange(len(testcases))
        width = 0.38

        if runtime:
            fig, ax = plt.subplots(figsize=(7, 4))
            b1 = ax.bar(x - width/2, data["dense_times"],   width, label="dense")
            b2 = ax.bar(x + width/2, data["coo_times"], width, label="COO")
        
            ax.set_xticks(x, testcases)
            ax.set_yscale('log')
            ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
            ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
            ax.legend()
            ax.set_xlabel("n")
            ax.set_ylabel("Time (ms)")
        
            plt.tight_layout()
            plt.show()

        if error:
            fig, ax = plt.subplots(figsize=(7, 4))
            b1 = ax.bar(x - width/2, data["dense_errors"],   width, label="dense")
            b2 = ax.bar(x + width/2, data["coo_errors"], width, label="COO")
        
            ax.set_xticks(x, testcases)
            ax.bar_label(b1, fmt="%.2fe-6", padding=2, fontsize=8)
            ax.bar_label(b2, fmt="%.2fe-6", padding=2, fontsize=8)
            ax.legend()
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: rf"${x:.1f} \times 10^{{-6}}$"))
            ax.set_xlabel("n")
            ax.set_ylabel("Error")
        
            plt.tight_layout()
            plt.show()


    # Returns the dominant eigenvalue paired with its normalized eigenvector
    def power_method_dense(self, mtrx: np.ndarray[np.ndarray[float]], v: np.ndarray[float] | None = None,
                            error: float = 10**(-4), max_iter: int = 10**4,
                            history: dict[str, list[float]] | None = None) -> tuple[float, np.ndarray[float]] | None:
        if linalg.norm(mtrx) == 0:
                    return None

        if v is None:
            v = np.full(len(mtrx), 1 / math.sqrt(len(mtrx)))

        for _ in range(max_iter):
            start = time.perf_counter_ns()

            av = self.dense_matvec(mtrx, v)

            # Rayleigh quotient. v is already normalized so the denominator v @ v is 1, and av is
            # already in hand, making the eigenvalue estimate O(n) against the O(n^2) product above
            lam = v @ av

            v_next = (av) / linalg.norm(av)
            converged = linalg.norm(v_next - v) < error

            self.record(history, "eigenvalues", lam, time.perf_counter_ns() - start)

            if converged:
                return lam, v_next

            v = v_next

        return lam, v

    # Returns the dominant eigenvalue paired with its normalized eigenvector
    def power_method_coo(self, mtrx: np.ndarray[np.ndarray[float]], v: np.ndarray[float] | None = None,
                           error: float = 10**(-4), max_iter: int = 10**4,
                           history: dict[str, list[float]] | None = None) -> tuple[float, np.ndarray[float]] | None:
        if linalg.norm(mtrx) == 0:
            return None

        if v is None:
            v = np.full(len(mtrx), 1 / math.sqrt(len(mtrx)))

        n = len(mtrx)
        coords, vals = self.get_coo(mtrx)

        for _ in range(max_iter):
            start = time.perf_counter_ns()

            av = self.coo_matvec(coords, vals, v, n)

            lam = v @ av

            v_next = av / linalg.norm(av)
            converged = linalg.norm(v_next - v) < error

            self.record(history, "eigenvalues", lam, time.perf_counter_ns() - start)

            if converged:
                return lam, v_next

            v = v_next

        return lam, v


    # Runs both implementations against the same matrix and returns their per-iteration eigenvalue
    # estimate alongside the runtime accumulated through that iteration. Single run, for the same
    # reason as get_iteration_data_jacobian.
    def get_iteration_data_power(self, n: int) -> dict[str, list[float]]:
        mtrx = self.construct_random_sparse(n)

        dense = {"eigenvalues": [], "times": []}
        coo = {"eigenvalues": [], "times": []}

        self.power_method_dense(mtrx, history=dense)
        self.power_method_coo(mtrx, history=coo)

        return { "dense_times" : dense["times"], "coo_times" : coo["times"],
                 "dense_eigenvalues" : dense["eigenvalues"], "coo_eigenvalues" : coo["eigenvalues"]}


    # Estimated eigenvalue and accumulated runtime per iteration, dense against COO
    def compare_power(self, n: int, *, eigenvalue: bool = True, runtime: bool = True) -> None:
        data = self.get_iteration_data_power(n)

        dense_k = np.arange(1, len(data["dense_eigenvalues"]) + 1)
        coo_k = np.arange(1, len(data["coo_eigenvalues"]) + 1)

        if eigenvalue:
            fig, ax = plt.subplots(figsize=(7, 4))

            # As with Jacobi the two implementations produce identical iterates, so these coincide
            ax.plot(dense_k, data["dense_eigenvalues"], label="dense", marker="o", markersize=4)
            ax.plot(coo_k, data["coo_eigenvalues"], label="COO", marker="x", markersize=6, linestyle="--")

            ax.legend()
            ax.set_xlabel("Iteration k")
            ax.set_ylabel(r"Estimated eigenvalue $\lambda^{(k)}$")

            plt.tight_layout()
            plt.show()

        if runtime:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(dense_k, data["dense_times"], label="dense", marker="o", markersize=4)
            ax.plot(coo_k, data["coo_times"], label="COO", marker="x", markersize=6, linestyle="--")

            ax.set_yscale('log')
            ax.legend()
            ax.set_xlabel("Iteration k")
            ax.set_ylabel("Accumulated runtime (ms)")

            plt.tight_layout()
            plt.show()


    def get_scaling_data_power(self, testcases: tuple[int], runs: int) -> dict[str, list[float]]:
        data = { "dense_times" : [], "coo_times" : [], "dense_errors" : [], "coo_errors" : []}

        for n in testcases:
            coo_times, dense_times, coo_errors, dense_errors = [], [], [], []

            for _ in range(runs):
                mtrx = self.construct_random_sparse(n)

                dense_start = time.perf_counter_ns()
                lam, v = self.power_method_dense(mtrx)
                dense_times.append(time.perf_counter_ns() - dense_start)
                dense_errors.append(linalg.norm(mtrx @ v - lam * v))

                coo_start = time.perf_counter_ns()
                lam, v = self.power_method_coo(mtrx)
                coo_times.append(time.perf_counter_ns() - coo_start)
                coo_errors.append(linalg.norm(mtrx @ v - lam * v))

            data["dense_times"].append((sum(dense_times) / runs) / 10**6); data["coo_times"].append((sum(coo_times) / runs) / 10**6)
            data["dense_errors"].append((sum(dense_errors) / runs) * 10**3); data["coo_errors"].append((sum(coo_errors) / runs) * 10**3)

        return data


    # How the dense/COO gap widens with n. Not required by the assignment
    def scaling_power(self, testcases: tuple[int], runs: int = 10, *,
                         error: bool = True, runtime: bool = True) -> None:
        data = self.get_scaling_data_power(testcases, runs)

        x = np.arange(len(testcases))
        width = 0.38

        if runtime:
            fig, ax = plt.subplots(figsize=(7, 4))
            b1 = ax.bar(x - width/2, data["dense_times"],   width, label="dense")
            b2 = ax.bar(x + width/2, data["coo_times"], width, label="COO")

            ax.set_xticks(x, testcases)
            ax.set_yscale('log')
            ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
            ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
            ax.legend()
            ax.set_xlabel("n")
            ax.set_ylabel("Time (ms)")

            plt.tight_layout()
            plt.show()

        if error:
            fig, ax = plt.subplots(figsize=(7, 4))
            b1 = ax.bar(x - width/2, data["dense_errors"],   width, label="dense")
            b2 = ax.bar(x + width/2, data["coo_errors"], width, label="COO")

            ax.set_xticks(x, testcases)
            ax.bar_label(b1, fmt="%.2fe-3", padding=2, fontsize=8)
            ax.bar_label(b2, fmt="%.2fe-3", padding=2, fontsize=8)
            ax.legend()
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: rf"${x:.1f} \times 10^{{-3}}$"))
            ax.set_xlabel("n")
            ax.set_ylabel("Error")

            plt.tight_layout()
            plt.show()


matricies = Matricies()

# Solves the assigned system once and reports the numbers the write-up needs. Kept separate from the
# comparison plots, which build their own matrix, so the printed figures describe a single solve.
def report(n: int) -> None:
    mtrx = matricies.construct_random_sparse(n)
    b = np.ones(n)

    jacobi = {"errors": [], "times": []}
    power = {"eigenvalues": [], "times": []}

    x = matricies.jacobi_coo(mtrx, b, history=jacobi)
    lam, v = matricies.power_method_coo(mtrx, history=power)

    nonzeros = np.count_nonzero(mtrx) - n

    print(f"N = {n}, nonzero off-diagonal entries per row = {nonzeros / n:.2f} (expected {n**(1/3):.2f})")
    print()
    print(f"Jacobi: converged in {len(jacobi['errors'])} iterations, {jacobi['times'][-1]:.2f} ms")
    print(f"  final ||Ax - b||_2 = {jacobi['errors'][-1]:.3e}")
    print(f"  x range [{x.min():.6f}, {x.max():.6f}], mean {x.mean():.6f}")
    print()
    print(f"Power method: converged in {len(power['eigenvalues'])} iterations, {power['times'][-1]:.2f} ms")
    print(f"  dominant eigenvalue = {lam:.6f}")
    print(f"  ||Av - (lambda)v||_2 = {linalg.norm(mtrx @ v - lam * v):.3e}, ||v||_2 = {linalg.norm(v):.6f}")
    print(f"  v range [{v.min():.6f}, {v.max():.6f}]")
    print()


def main():
    n = 2**10

    report(n)

    matricies.compare_jacobian(n)

    matricies.compare_power(n)

    testcases = (3, 10, 20, 50, 100)
    runs = 15

    matricies.scaling_jacobian(testcases, runs)
    matricies.scaling_power(testcases, runs)

if __name__ == "__main__":
    main()