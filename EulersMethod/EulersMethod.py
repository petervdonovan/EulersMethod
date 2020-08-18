import math
import time
f = math.exp
fprime = lambda x, y: y

# Surprisingly, v1 has continued to perform the highest, 
# even after the implementation of v2 and v3 as extensions 
# of the technique used in v1)

def test(approximator, f, fprime, x0, y0, x1, steps):
    """
    Returns the error of the approximation and the time it took to calculate it.
    approximator must be some function of fprime, x0, y0, and x1.
    f is a function of x.
    fprime is the first derivative of f, in terms of x and y.
    x0 and y0 are the initial point.
    x1 is the value of x for which the function is to be evaluated.
    steps is the number of steps allowed for the calculation.
    """
    t0 = time.time()
    y1 = approximator(fprime, x0, y0, x1, steps)
    t1 = time.time()
    return (y1 - f(x1), t1 - t0)

def eulers_method_maker(take_step):
    """
    Returns a variation on Euler's method based on a version
    of the process of taking one step.
    """
    def eulers_method(fprime, x0, y0, x1, steps):
        """
        Returns an approximation of the value of f at x1.
        fprime is the first derivative of f, in terms of x and y.
        x0 and y0 are the initial point.
        x1 is the value of x for which the function is to be evaluated.
        steps is the number of steps allowed for the calculation.
        """
        x = x0
        y = y0
        step = (x1 - x0) / steps
        if x0 < x1:
            while x < x1 - step / 2:
                x, y = take_step(x, y, fprime, step)
        else:
            while x > x1 - step / 2:
                x, y = take_step(x, y, fprime, step)
        return y
    return eulers_method
basic = lambda x, y, fprime, step: (x + step, y + fprime(x, y) * step)
v0 = eulers_method_maker(basic)
def endpoint_mean_slope(x_n, y_n, fprime, step, negligible_proportion_slope_diff=0.0001):
    """
    Repeatedly recalculates the slope between x and x + step as the mean of the 
    slopes at the endpoints, and then uses the new slope to get a new endpoint.
    The process ends when the slope ceases to change significantly.
    """
    # The following line is needed to maintain accuracy as steps become very small
    negligible_proportion_slope_diff = min(step / 2, negligible_proportion_slope_diff)
    fprime_n = fprime(x_n, y_n)
    x_n_plus_one = x_n + step
    y_n_plus_one = y_n + fprime_n * step
    fprime_n_plus_one = lambda: fprime(x_n_plus_one, y_n_plus_one)
    mean_endpoints = lambda: (fprime_n_plus_one() + fprime_n) / 2
    mean_slope = lambda: (y_n_plus_one - y_n) / (step)
    while abs(
        (mean_endpoints() - mean_slope()) / mean_slope()
        ) > negligible_proportion_slope_diff:
        y_n_plus_one = y_n + mean_endpoints() * step
    return x_n_plus_one, y_n_plus_one
v1 = eulers_method_maker(endpoint_mean_slope)
def smart_step_size(take_step):
    """
    Returns a modified version of the step method passed that uses a
    variable number of iterations, depending on the magnitude of the slope change
    across the interval of one step.
    """
    def stepper(x_n, y_n, fprime, step, iterations_per_unit_slope_diff=64):
        tangent_line_endpoint = basic(x_n, y_n, fprime, step)
        abs_slope_diff = abs(
            fprime(x_n, y_n) 
            - fprime(
                tangent_line_endpoint[0],
                tangent_line_endpoint[1]
                )
            )
        iterations = math.floor(abs_slope_diff * iterations_per_unit_slope_diff)
        iterations = max(1, iterations)
        x0 = x_n
        y0 = y_n
        x1 = x0 + step
        return x1, eulers_method_maker(take_step)(fprime, x0, y0, x1, iterations)
    return stepper
v2 = eulers_method_maker(smart_step_size(endpoint_mean_slope))

def run_tests():
    n = 4
    previous_error = None
    while n <= 1024:
        result = test(approximator, f, fprime, 0, 1, 1, n)
        print(n, "steps result:", result)
        if previous_error is not None:
            print(
                "Error reduction:", 
                round(
                    (abs(previous_error) - abs(result[0])) / abs(previous_error) * 100, 
                    1
                    ),
                "%"
                )
        previous_error = result[0]
        n *= 2

def v3(fprime, x0, y0, x1, steps, iterations=4):
    """
    Repeatedly performs the basic step over a list of x values,
    refining the list each time so that the spacing of the different
    x values decreases as estimated second derivative increases.
    Then, performs Euler's method using the endpoint_mean_slope step method.
    """
    steps_per_iteration = math.floor(steps / iterations)
    # Double number of steps in initial iteration by halving initial size and doubling # of steps
    initial_step_size = ((x1 - x0) / steps_per_iteration) / 2
    xlist = [x0 + initial_step_size * i for i in range(2*steps_per_iteration + 1)]
    ylist = []
    fprimeslist = []
    for i in range(iterations):
        ylist = [y0]
        fprimeslist = [fprime(x0, y0)]
        y = y0
        for i in range(len(xlist) - 1):
            ylist.append(basic(xlist[i], ylist[i], fprime, xlist[i+1])[1])
            fprimeslist.append((ylist[-1] - ylist[-2]) / (xlist[i+1] - xlist[i]))
        abs_delta_fprimes = [
            (i, abs(fprimeslist[i+1] - fprimeslist[i])) 
            for i in range(len(fprimeslist)-1)
            ]
        abs_delta_fprimes.sort(key=lambda tup: tup[1])
        large_second_derivatives = abs_delta_fprimes[-steps_per_iteration:]
        large_second_derivatives.sort(key=lambda tup: tup[0], reverse=True)
        for i, _ in large_second_derivatives:
            xlist.insert(i+1, (xlist[i] + xlist[i+1]) / 2)
    y = y0
    for i in range(len(xlist) - 1):
        y = basic(xlist[i], y, fprime, xlist[i+1] - xlist[i])[1]
    return y

approximator = v0
run_tests()
print()

approximator = v1
run_tests()
print()

approximator = v3
run_tests()