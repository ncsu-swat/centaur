
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nextafter_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays, same shape
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 1.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D float64 arrays with positive and negative numbers
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float64)
    y = np.array([-2.0, -1.0, 2.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D float32 arrays
    x = np.ones((3, 3), dtype=np.float32)
    y = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Broadcasting (2D and 1D)
    x = np.zeros((2, 4), dtype=np.float32)
    y = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Scalar 0D arrays
    x = np.array(1.0, dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D float32 arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Float16 arrays
    x = np.array([-0.0, 0.0, 100.0], dtype=np.float16)
    y = np.array([1.0, -1.0, np.inf], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Special floating point values (inf, -inf)
    x = np.array([np.inf, -np.inf, 1.0], dtype=np.float32)
    y = np.array([0.0, 0.0, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D broadcast compatible arrays
    x = np.ones((2, 1, 3, 1), dtype=np.float32)
    y = np.zeros((1, 4, 1, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Empty arrays
    x = np.empty((0, 2), dtype=np.float64)
    y = np.empty((0, 2), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Subnormal edge cases with float32
    x = np.array([0.0, -0.0, 1e-40], dtype=np.float32)
    y = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.nextafter_1"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nextafter_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nextafter_1'.")


check_valid('jax.numpy.nextafter', generated_inputs['jax.numpy.nextafter_1'], lib="jax", suffix=1)
