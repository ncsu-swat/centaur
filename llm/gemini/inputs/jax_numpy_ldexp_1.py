
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ldexp_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays with float32 and int32
    x1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: Negative values in mantissa and exponent
    x1 = np.array([-1.5, 2.5, -0.75, 0.125], dtype=np.float32)
    x2 = np.array([-2, -1, 0, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D arrays of same shape
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randint(-5, 6, size=(3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting (3, 1) with (1, 4)
    x1 = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    x2 = np.array([[0, 1, 2, 3]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D array with float64 and int64
    x1 = np.random.randn(2, 3, 2).astype(np.float64)
    x2 = np.random.randint(-10, 10, size=(2, 3, 2), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalars (0-dimensional arrays)
    x1 = np.array(3.5, dtype=np.float32)
    x2 = np.array(4, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Broadcast 0D x2 across 2D x1
    x1 = np.ones((2, 5), dtype=np.float32)
    x2 = np.array(-3, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: float16 precision and int16 exponent
    x1 = np.array([0.5, 1.5, -2.5], dtype=np.float16)
    x2 = np.array([2, -1, 3], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 4D tensor broadcasting with (1, 1, 1, 1)
    x1 = np.random.randn(2, 2, 3, 4).astype(np.float32)
    x2 = np.array([[[[2]]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: int8 exponents and broadcasting 1D with 2D
    x1 = np.array([[0.5, 0.25], [-0.5, -0.25]], dtype=np.float32)
    x2 = np.array([1, 2], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Empty array edge case
    x1 = np.array([], dtype=np.float32)
    x2 = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.ldexp_1"] = ldexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ldexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ldexp_1'.")


check_valid('jax.numpy.ldexp', generated_inputs['jax.numpy.ldexp_1'], lib="jax", suffix=1)
