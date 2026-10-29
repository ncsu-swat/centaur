
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def heaviside_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x1 = np.array([-1.5, 0.0, 2.3, -0.1, 0.0, 5.0], dtype=np.float32)
    x2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float64 arrays with zeros, positives, and negatives
    x1 = np.array([[-2.0, 0.0, 3.0], [5.0, -1.0, 0.0]], dtype=np.float64)
    x2 = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 0D arrays (scalars wrapped in np.array)
    x1 = np.array(0.0, dtype=np.float32)
    x2 = np.array(0.75, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting 2D array and 1D array
    x1 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]], dtype=np.float32)
    x2 = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: Integer arrays (int32)
    x1 = np.array([-5, -1, 0, 0, 1, 10], dtype=np.int32)
    x2 = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 3D float32 arrays
    x1 = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    x2 = np.full((2, 3, 4), 0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Broadcasting with different dimensions (3, 1) and (1, 4)
    x1 = np.array([[-2.0], [0.0], [1.5]], dtype=np.float32)
    x2 = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Float16 arrays
    x1 = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    x2 = np.array([0.5, 0.5, 0.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 4D arrays
    x1 = np.zeros((2, 2, 2, 2), dtype=np.float32)
    x2 = np.ones((2, 2, 2, 2), dtype=np.float32) * 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Array with inf and -inf values
    x1 = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf], dtype=np.float32)
    x2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Mixed integer and float types via broadcasting (0D x2 with 2D x1)
    x1 = np.array([[10, -10], [0, 0]], dtype=np.int64)
    x2 = np.array(0.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.heaviside_1"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_1'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_1'], lib="jax", suffix=1)
