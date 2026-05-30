
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def func1(x, y):
    return x + y

def func2(x):
    return x ** 2

def func3(x, y):
    return x * y

def func4(x, y, z):
    return x + y + z

def func5(x, y):
    return x - y

def func6(x):
    return x + 1

def func7(x, y):
    return x % (y + 1)

def func8(x, y, z):
    return x * y * z

def func9(x):
    return x * 0.5

def func10(x, y):
    return x < y

def fromfunction_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "function": func1,
        "shape": (3, 3),
        "dtype": np.int32
    })

    # Input 2
    list_of_inputs.append({
        "function": func2,
        "shape": (5,),
        "dtype": np.float32
    })

    # Input 3
    list_of_inputs.append({
        "function": func3,
        "shape": (4, 4),
        "dtype": np.int32
    })

    # Input 4
    list_of_inputs.append({
        "function": func4,
        "shape": (2, 3, 4),
        "dtype": np.float32
    })

    # Input 5
    list_of_inputs.append({
        "function": func5,
        "shape": (5, 2),
        "dtype": np.float64
    })

    # Input 6
    list_of_inputs.append({
        "function": func6,
        "shape": (10,),
        "dtype": np.int16
    })

    # Input 7
    list_of_inputs.append({
        "function": func7,
        "shape": (3, 5),
        "dtype": np.int32
    })

    # Input 8
    list_of_inputs.append({
        "function": func8,
        "shape": (2, 2, 2),
        "dtype": np.float32
    })

    # Input 9
    list_of_inputs.append({
        "function": func9,
        "shape": (8,),
        "dtype": np.float32
    })

    # Input 10
    list_of_inputs.append({
        "function": func10,
        "shape": (4, 4),
        "dtype": np.int32
    })

    return list_of_inputs

generated_inputs["jax.numpy.fromfunction_1"] = fromfunction_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fromfunction_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fromfunction_1'.")


check_valid('jax.numpy.fromfunction', generated_inputs['jax.numpy.fromfunction_1'], lib="jax", suffix=1)
