
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values with float weights
    x = np.array([1, 1, 2, 3, 3, 3], dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 0,
        "length": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative integers (clipped to 0 in JAX)
    x = np.array([-1, -2, 1, 2, -1], dtype=np.int32)
    weights = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 0,
        "length": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Large scale inputs with integer weights
    x = np.random.randint(0, 10, size=100).astype(np.int32)
    weights = np.ones(100, dtype=np.int32)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 5,
        "length": 12
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Out of bounds values (larger than length + 1 are dropped)
    x = np.array([1, 2, 10, 15, 2], dtype=np.int32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 0,
        "length": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Minlength larger than length
    x = np.array([1, 2, 2], dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 8,
        "length": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: High precision float weights
    x = np.array([0, 0, 1, 1], dtype=np.int64)
    weights = np.array([0.123456789, 0.987654321, 1.111111111, 2.222222222], dtype=np.float64)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 2,
        "length": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element arrays
    x = np.array([4], dtype=np.int32)
    weights = np.array([10.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 1,
        "length": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All zeros input
    x = np.zeros(10, dtype=np.int32)
    weights = np.linspace(0, 1, 10, dtype=np.float32)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 2,
        "length": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex negative and positive mix
    x = np.array([-5, 0, 5, -10, 10], dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 0,
        "length": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger length limitation with elements at the boundary
    x = np.array([18, 19, 20, 21, 2], dtype=np.int32)
    weights = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "weights": weights,
        "minlength": 0,
        "length": 20
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.bincount"] = bincount_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bincount'.")


check_valid('jax.numpy.bincount', generated_inputs['jax.numpy.bincount'], lib="jax", suffix=0)
