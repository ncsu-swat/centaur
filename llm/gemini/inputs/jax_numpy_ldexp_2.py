
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ldexp_inputs():
    list_of_inputs = []

    # 1D float32 input with positive integer exponent
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = 2
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # 2D float64 input with negative integer exponent
    x1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    x2 = -1
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # 3D float32 input with zero exponent
    x1 = np.ones((2, 2, 2), dtype=np.float32) * -5.0
    x2 = 0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # 1D float16 input with larger integer exponent
    x1 = np.array([0.1, 0.5, 0.9], dtype=np.float16)
    x2 = 10
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # 4D random float32 input
    x1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x2 = 3
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # 0D float32 input (scalar tensor)
    x1 = np.array(3.14, dtype=np.float32)
    x2 = -2
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # 2D float32 input with special values
    x1 = np.array([[np.inf, -np.inf], [np.nan, 0.0]], dtype=np.float32)
    x2 = 1
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # 1D float64 input
    x1 = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    x2 = 5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # 2D uniform float32 input with negative exponent
    x1 = np.random.uniform(-10, 10, (3, 3)).astype(np.float32)
    x2 = -5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # 5D float32 input
    x1 = np.random.randn(1, 2, 1, 2, 1).astype(np.float32)
    x2 = 4
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.ldexp_2"] = ldexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ldexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ldexp_2'.")


check_valid('jax.numpy.ldexp', generated_inputs['jax.numpy.ldexp_2'], lib="jax", suffix=2)
