
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cbrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 0D (scalar) float64 array
    x = np.array(64.0, dtype=np.float64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 2D float32 array (matrix)
    x = np.array([[-125.0, 8.0], [27.0, -64.0]], dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3D float64 array with uniform values
    x = np.random.uniform(-100.0, 100.0, size=(2, 3, 4)).astype(np.float64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 1D float16 array
    x = np.array([-8.0, -1.0, 0.0, 1.0, 8.0], dtype=np.float16)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 1D float32 array with fractions
    x = np.array([0.125, -0.008, 0.001, -0.027], dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 2D float64 array
    x = np.array([[-1000.0, 343.0], [216.0, -729.0]], dtype=np.float64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: Large values in float32
    x = np.array([1e6, -1e6, 1e9, -1e9], dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: Tiny near-zero values in float64
    x = np.array([1e-12, -1e-12, 1e-20, -1e-20], dtype=np.float64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 5D float32 array
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32) * -8.0
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 12: Empty 1D float32 array
    x = np.array([], dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.cbrt_2"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cbrt_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cbrt_2'.")


check_valid('jax.lax.cbrt', generated_inputs['jax.lax.cbrt_2'], lib="jax", suffix=2)
