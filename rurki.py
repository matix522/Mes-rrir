import numpy as np
import scipy as sp
from scipy.misc import derivative
from scipy.integrate import quad
from numpy.linalg import solve
from matplotlib import pyplot as plt
from functools import reduce

# ODE funtions
function_a = lambda x: 1
function_b = lambda x: 0
function_c = lambda x: 1
function_f = lambda x: 0

# ODE parameters
begin_val = -3 * np.pi
end_val = 4 * np.pi
range_count = int(35)

u1 = np.cos(end_val)
beta = 0
gamma = 0

# helper variables
u_vector = np.zeros(range_count + 1)
functions = []
breaks = np.linspace(begin_val, end_val, range_count)
limit = 250


def get_shape_function(begin: int, end: int, range_count: int, number: int):
    return lambda x: max(0, 1 - abs(range_count / (end - begin) * (-begin + x - number * (end - begin) / range_count)))


form_a = lambda x, u, v: function_a(x) * derivative(u, x, dx=1e-6) * derivative(v, x, dx=1e-6)

form_b = lambda x, u, v: function_b(x) * derivative(u, x, dx=1e-6) * v(x)

form_c = lambda x, u, v: function_c(x) * u(x) * v(x)


def b_form_value(u, v):
    return (
        -beta * u(begin_val) * v(begin_val)
        - quad(form_a, begin_val, end_val, (u, v), points=breaks, limit=limit)[0]
        + quad(form_b, begin_val, end_val, (u, v), points=breaks, limit=limit)[0]
        + quad(form_c, begin_val, end_val, (u, v), points=breaks, limit=limit)[0]
    )


def l_form_value(v):
    form_f = lambda x, v: function_f(x) * v(x)
    return quad(form_f, begin_val, end_val, v, points=breaks, limit=limit)[0] - gamma * v(begin_val)


def result(x):
    val = 0
    for i in range(0, range_count + 1):
        val += u_vector[i] * get_shape_function(begin_val, end_val, range_count, i)(x)
    return val


def test():
    x_values = np.linspace(begin_val, end_val, 1000)
    for i in range(0, range_count + 1):
        y_values = list(map(get_shape_function(begin_val, end_val, range_count, i), x_values))
        plt.plot(x_values, y_values)

    plt.show()


def run():
    b_matrix = np.zeros((range_count, range_count))
    for i in range(0, range_count):
        for j in range(0, range_count):
            e_u = get_shape_function(begin_val, end_val, range_count, i)
            e_v = get_shape_function(begin_val, end_val, range_count, j)
            b_matrix[i, j] = b_form_value(e_v, e_u)

    l_vector = np.zeros(shape=range_count)
    for i in range(0, range_count):
        e_u = lambda x: u1 * get_shape_function(begin_val, end_val, range_count, range_count)(x)
        e_v = get_shape_function(begin_val, end_val, range_count, i)
        l_vector[i] = l_form_value(e_v) - b_form_value(e_u, e_v)

    global u_vector

    print(b_matrix)
    print(l_vector)

    u_vector = solve(b_matrix, l_vector)
    u_vector = np.insert(u_vector, np.size(u_vector), u1)

    print(u_vector)

    x_values = np.linspace(begin_val, end_val, 100)
    y_values = list(map(result, x_values))

    print(x_values)
    print(y_values)

    # plt.ylim(0, 3)
    # plt.xlim(0, 1)

    plt.plot(x_values, y_values)
    plt.plot(x_values, list(map(np.cos, x_values)))
    plt.show()


test()
run()
