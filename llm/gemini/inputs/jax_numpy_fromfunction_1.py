
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class StringFunction(str):

    def __new__(cls, expr):
        obj = super().__new__(cls, expr)
        obj._fn = eval(expr)
        return obj

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __deepcopy__(self, memo):
        return StringFunction(str(self))


def fromfunction_inputs():
    list_of_inputs = []

    # Input 1: 2D integer addition
    input_dict = {
        "function": StringFunction("lambda i, j: i + j"),
        "shape": (3, 3),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float square
    input_dict = {
        "function": StringFunction("lambda i: i ** 2"),
        "shape": (10,),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D linear combination
    input_dict = {
        "function": StringFunction("lambda i, j: i * 2 + j"),
        "shape": (4, 4),
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D sum of coordinates
    input_dict = {
        "function": StringFunction("lambda i, j, k: i + j + k"),
        "shape": (2, 3, 4),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D multiplication table
    input_dict = {
        "function": StringFunction("lambda i, j: i * j"),
        "shape": (5, 6),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D scaling
    input_dict = {
        "function": StringFunction("lambda i: i * 2.5"),
        "shape": (5,),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D coordinates
    input_dict = {
        "function": StringFunction("lambda i, j, k, l: i + j + k + l"),
        "shape": (2, 2, 2, 2),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float subtraction
    input_dict = {
        "function": StringFunction("lambda i, j: i - j"),
        "shape": (1, 8),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D single-element shape
    input_dict = {
        "function": StringFunction("lambda i: i * 0"),
        "shape": (1,),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D non-square float operation
    input_dict = {
        "function": StringFunction("lambda x, y, z: (x + 1.0) * (y + 2.0) + z"),
        "shape": (3, 1, 4),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D integer products
    input_dict = {
        "function": StringFunction("lambda i, j: (i + 1) * (j + 1)"),
        "shape": (6, 6),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
