
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isneginf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with inf, -inf, nan, and finite values
    x = np.array([-np.inf, 0.0, np.inf, np.nan, -10.0], dtype=np.float32)
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    # Input 2: 2D float64 array
    x = np.array([[-np.inf, 1.0], [np.inf, -2.0]], dtype=np.float64)
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    # Input 3: 3D float16 array
    x = np.array([[[-np.inf], [3.0]], [[np.nan], [-np.inf]]], dtype=np.float16)
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    # Input 4: Scalar-like 0D float32 array
    x = np.array(-np.inf, dtype=np.float32)
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    # Input 5: 1D float64 array, all finite
    x = np.linspace(-10, 10, 5, dtype=np.float64)
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    # Input 6: 4D float32 array
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    x[0, 1, 0, 1] = -np.inf
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    # Input 7: 1D int32 array
    x = np.array([-1, 0, 1, 2], dtype=np.int32)
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    # Input 8: 2D float32 random array with one negative infinity
    x = np.random.randn(3, 3).astype(np.float32)
    x[1, 1] = -np.inf
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    # Input 9: 3D float64 array containing only negative infinities
    x = np.ones((2, 3, 4), dtype=np.float64) * -np.inf
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    # Input 10: 5D float32 array with singleton dimensions
    x = np.zeros((1, 2, 1, 3, 1), dtype=np.float32)
    x[0, 1, 0, 2, 0] = -np.inf
    out = None
    list_of_inputs.append(copy.deepcopy({"x": x, "out": out}))

    return list_of_inputs

generated_inputs["jax.numpy.isneginf_1"] = isneginf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isneginf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isneginf_1'.")


check_valid('jax.numpy.isneginf', generated_inputs['jax.numpy.isneginf_1'], lib="jax", suffix=1)
