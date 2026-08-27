import matplotlib.pyplot as plt


class LCG:
    def __init__(self, x_0: float = 1):
        self.x_0 = x_0 

    # Basic LCG equation
    def lcg(self, m: int = 2**16 + 1, a: int = 75, c: int = 1, 
            x_0: float | None = None, n: int = 1, normalize: bool = True) -> float: 
        
        if x_0 is None:
            x_0 = self.x_0
            iterate = True
        else:
            iterate = False

        if any((m <= 0, a <= 0, a >= m, c < 0, c >= m)):
            raise ValueError("Invalid argumets, make sure: m > 0,"
                             " 0 < a < m, and 0 <= c < m")

        if n == 0:
            return x_0
        
        x_n = (a * self.lcg(m, a, c, x_0, n - 1) + c) % m

        if iterate:
            self.x_0 = x_n

        return x_n / m if normalize else x_n

    # Uses LCG to generate a list of n pseudorandom numbers on [0, 1) with increment 
    # 0.01 using an LCG where m = 1000, c = 1, a = 999, and x_0 = 0.3
    def lcg_rand(self, n: int, a: float = 999, x_0: float = 0.3) -> list[float]: 

        m = 1_000
        c = 1
        x = x_0

        nums = []

        for i in range(n):
            x = self.lcg(m, a, c, x, 1, False)
            nums.append((x // 10) / 100)
        
        return nums
    

    # Plots distribution of the above func
    def histogram(self, n: int, m: int = 2**16 + 1, a: int = 75, c: int = 1, 
            x_0: float | None = None, normalize: bool = True) -> None:

        nums = [self.lcg(m, a, c) for _ in range(n + 1)]

        plt.hist(nums, bins=20, color='skyblue', edgecolor='black', range = (0, 1))

        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('LCG Distribution')

        plt.show()

    # Plots a scatter of Y_i against Y_{i - 1}
    def scatterplot(self, n: int, m: int = 2**16 + 1, a: int = 75, c: int = 1, 
            x_0: float | None = None, normalize: bool = True) -> None:

        nums = [self.lcg(m, a, c) for _ in range(n + 1)]
        ys = nums[1:]
        nums.pop()
        y_minus = nums

        plt.scatter(y_minus, ys)

        plt.xlabel("Y_{i - 1}")
        plt.ylabel("Y_{i}")

        plt.show()

random = LCG(1)

def main():
    for i in range(100):
        print(random.lcg(m = 100, a = 99, c = 1, normalize=False))
    random.histogram(1000)
    random.scatterplot(10)

if __name__ == "__main__":
    main()