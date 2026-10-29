
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class CallableString(str):

    def __init__(self, expr):
        super().__init__()
        self.func = eval(expr)
        self.__name__ = getattr(self.func, "__name__", "callable_str")
        self.__qualname__ = getattr(self.func, "__qualname__", "callable_str")

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def fromfunction_inputs():
    list_of_inputs = []

    # Input 1: Basic addition, int32 dtype
    input_dict = {
        "function": CallableString("lambda i: i + 1"),
        "shape": int(5),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Linear scaling, float32 dtype
    input_dict = {
        "function": CallableString("lambda i: i * 2.0"),
        "shape": int(10),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Linear transformation with negative offset, int64 dtype
    input_dict = {
        "function": CallableString("lambda i: i * 3 - 5"),
        "shape": int(4),
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Fractional computation, float64 dtype
    input_dict = {
        "function": CallableString("lambda i: i * 0.5 + 1.0"),
        "shape": int(8),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Offset with uint32 dtype
    input_dict = {
        "function": CallableString("lambda i: i + 10"),
        "shape": int(6),
        "dtype": np.dtype(np.uint32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiplication, int16 dtype
    input_dict = {
        "function": CallableString("lambda i: i * 2"),
        "shape": int(3),
        "dtype": np.dtype(np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scaling with float16 dtype
    input_dict = {
        "function": CallableString("lambda i: i * 1.5"),
        "shape": int(7),
        "dtype": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Addition, int8 dtype
    input_dict = {
        "function": CallableString("lambda i: i + 2"),
        "shape": int(2),
        "dtype": np.dtype(np.int8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small addition, uint8 dtype
    input_dict = {
        "function": CallableString("lambda i: i + 3"),
        "shape": int(12),
        "dtype": np.dtype(np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Quadratic expression, int32 dtype
    input_dict = {
        "function": CallableString("lambda i: i * i"),
        "shape": int(16),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Scaling, float32 dtype
    input_dict = {
        "function": CallableString("lambda i: i * 4.0"),
        "shape": int(20),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fromfunction_2"] = fromfunction_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fromfunction_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fromfunction_2'.")


check_valid('jax.numpy.fromfunction', generated_inputs['jax.numpy.fromfunction_2'], lib="jax", suffix=2)
