
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nextafter_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 scalars
    x1 = np.array(1.0, dtype=np.float32)
    x2 = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D float32 arrays with mixed signs
    x1 = np.array([0.0, -1.0, 1.0, -0.0], dtype=np.float32)
    x2 = np.array([1.0, -2.0, 0.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float64 arrays
    x1 = np.zeros((3, 3), dtype=np.float64)
    x2 = np.ones((3, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 1D float16 arrays with infinities
    x1 = np.array([0.0, np.inf, -np.inf, 1.0], dtype=np.float16)
    x2 = np.array([-np.inf, 0.0, 0.0, np.inf], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 2D float32 arrays with rank-matched broadcasting
    x1 = np.ones((1, 4), dtype=np.float32)
    x2 = np.full((3, 4), 2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 3D float32 arrays with random values
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float64 arrays
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float64)
    x2 = np.random.randn(2, 2, 2, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 2D float32 with negative values moving towards positive infinity
    x1 = -np.ones((2, 5), dtype=np.float32)
    x2 = np.full((2, 5), np.inf, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 1D float64 arrays with small numbers near zero
    x1 = np.array([1e-300, -1e-300, 0.0], dtype=np.float64)
    x2 = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 3D float16 arrays with rank-matched broadcasting
    x1 = np.zeros((2, 1, 3), dtype=np.float16)
    x2 = np.ones((2, 4, 3), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.lax.nextafter_1"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.nextafter_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.nextafter_1'.")


check_valid('jax.lax.nextafter', generated_inputs['jax.lax.nextafter_1'], lib="jax", suffix=1)
