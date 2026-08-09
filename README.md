## Numerical Methods

The following is a collection of numerical and statistical method implimentations in Python (/src) and 
Jupyter notebooks (/notebooks) with writeups/benchmarks/visualizations for each category of numerical
methods, under the same name as their underlying source files. The notebooks have quite a bit of code,
but it is all for benchmarking/visualizing, all actual implementations are in the source files (though
most notebooks contain some psuedocode).

Below is a list of each file/notebook and what they go over:

* `fixed_point` - Solving fixed point problems of the form $x = g(x)$, and limitations.
* `integrals_roots` - Various methods for approximating integrals (midpoint, trapazoid, 
Monte Carlo, TODO: Simpson's); Newton's method and bisection (binary) method for finding roots.
* `interpolations` - Piecewise and polynomial interpolations of functions,
chebyshev distribution sampling.
* `minimization` - Gradient descent (shown in 1 and 2 dimensions) and simmulated annealing.
* `pseudorandom_numbers` - Linear congruential generator implementation and limitations. TODO: add Mersenne 
Twister + block/stream cipher methods, briefly explain OS/system-based generators.
* `sampling` - Understanding energy functions, sampling from a distribution using inverse and MCMC 
methods (and limitations)
* `sparse_methods` - Implementation of coordinate matricies, Jacobian and Power methods for solving
linear problems and finding eigenvalues respectively.

My favorites are: sparse_methods and sampling (MCMC + cdf).

Insperation + sources for these implementations and writeups includes but is not limited to:

Numerical Analysis by Timothy Sauer ([pdf](https://eclass.aueb.gr/modules/document/file.php/MISC249/Sauer%20-%20Numerical%20Analysis%202e.pdf)) \
A Student's Guide to Baysian Statistics by Ben Labert ([pdf](https://sites.math.rutgers.edu/~zeilberg/EM20/Lambert.pdf)) ([videos](https://www.youtube.com/playlist?list=PLwJRxp3blEvZ8AKMXOy0fc0cqT61GsKCG)) (current) \
[3Blue1Brown](https://www.youtube.com/@3blue1brown) on YouTube \
[StudySession](https://www.youtube.com/@StudySessionYT) on YouTube \
[SISL](https://www.youtube.com/@SISLaboratory) on YouTube \
[Claude](claude.ai) by Anthropic