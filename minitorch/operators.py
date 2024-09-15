"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y:float) -> float:
    """Multiplies two numbers

    Args:
        x: float to be multiplied with y
        y: float to be multiplied with x

    Returns:
        x multiplied by y

    """
    return x * y


def id(x: float) -> float:
    """Identity function: returns the input unchanged

    Args:
        x: a float

    Returns:
        The input (float) unchanged
        
    """
    return x


def add(x: float, y: float) -> float:
    """Adds two floats

    Args:
        x: float to be summed with y
        y: float to be summed with x

    Returns:
        Sum of x and y
        
    """
    return x + y


def neg(x: float) -> float:
    """Negates a number

    Args:
        x: A float to be negated

    Returns:
        x negated
        
    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if first input is less than the second input

    Args:
        x: float
        y: float

    Returns:
        boolean specifying whether x is less than y

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks whether the two inputs are equal

    Args:
        x: float
        y: float

    Returns:
        Boolean specifying whether x and y are equal

    """
    return x == y


def max(x: float, y: float) -> float:
    """Calculates the maximum of the two inputs

    Args:
        x: float
        y: float

    Returns:
        (float) the maximum of x and y

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Returns whether x and y are with 0.01 of each other
    
    Args:
        x: float
        y: float
        
    Returns:
        (bool) whether x and y are close

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Returns the sigmoid function of the input

    Args:
        x: float

    Returns:
        (float) Sigmoid(x)

    """
    ans: float
    if x >= 0.:
        ans = 1. / (1. + math.exp(-x))
    else:
        ans = math.exp(x) / (1. + math.exp(x))
    return ans


def relu(x: float) -> float:
    """Applies the ReLU activation function to the input
    
    Args:
        x: float
        
    Returns:
        ReLU(x) (float)   

    """
    return x if x > 0. else 0.


def log(x: float) -> float:
    """Applies the natural logarithm to the input

    Args:
        x: float

    Returns:
        float: log(x)

    """
    return math.log(x)


def exp(x: float) -> float:
    """Applies the exponential function to the input

    Args:
        x: float

    Returns:
        float: exp(x)

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the reciprocal of the input

    Args:
        x: float
    
    Returns:
        (float) The reciprocal of x

    """
    return 1. / x


def log_back(x: float, y: float) -> float:
    """Returns the derivative of the log function of times another argument
    
    Args:
        x: take d/dx log 
        y: argmuent to multiple to d/dx log

    Returns:
        (float) the reciprocal of x times y

    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """Returns the derivative of the inverse function of the input times another arg
    
    Args:
        x: float

    Returns:
        (float) The derivative of the inverse function

    """
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Returns the derivative of ReLU of the first input times second argument

    Args:
        x: float
        y: float

    Returns:
        (float) derivative of ReLU on x times y

    """
    return y if x >= 0. else 0.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists



# TODO: Implement for Task 0.3.
