
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyadd_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays of different lengths
    a1 = np.array([1, 2, 3], dtype=np.int32)
    a2 = np.array([4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 2: 1D float32 arrays with negative values
    a1 = np.array([-2.5, 3.0], dtype=np.float32)
    a2 = np.array([1.0, -0.5, 4.2, -3.1], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 3: 1D float64 arrays of identical lengths
    a1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    a2 = np.array([-1.0, -2.0, 0.0, 5.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 4: 1D int64 arrays, scalar constant added to polynomial
    a1 = np.array([7], dtype=np.int64)
    a2 = np.array([3, -2, 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 5: 1D float32 arrays with different signs and sizes
    a1 = np.array([1.5, -3.2, 0.0, 4.1], dtype=np.float32)
    a2 = np.array([2.0, -1.1], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 6: 2D float32 arrays with identical shapes
    a1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a2 = np.array([[-1.0, 0.5], [2.0, -3.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 7: 2D int32 arrays with different leading dimensions
    a1 = np.array([[2, 3, 1]], dtype=np.int32)
    a2 = np.array([[5, 7, 3], [8, 2, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 8: 2D float64 arrays with different leading dimensions
    a1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    a2 = np.array([[7.0, 8.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 9: 3D float32 arrays of identical shapes
    a1 = np.ones((2, 2, 3), dtype=np.float32)
    a2 = np.zeros((2, 2, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 10: 1D int32 arrays with zero coefficients
    a1 = np.array([0, 0, 1, -5, 6], dtype=np.int32)
    a2 = np.array([1, 0, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 11: 1D float64 arrays with larger order
    a1 = np.array([1e-5, -2e-3, 3.5, -4.2, 5.1, -6.0], dtype=np.float64)
    a2 = np.array([10.0, -20.0, 30.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    return list_of_inputs


generated_inputs["jax.numpy.polyadd"] = polyadd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyadd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyadd'.")


check_valid('jax.numpy.polyadd', generated_inputs['jax.numpy.polyadd'], lib="jax", suffix=0)
