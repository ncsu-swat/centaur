
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def floor_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([-2.7, -1.5, -0.2, 0.0, 0.3, 1.5, 2.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array
    x = np.array(
        [[-3.8, -1.2, 0.0], [1.1, 4.9, -0.0], [-10.5, 12.3, 100.7]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array float32
    x = np.array(3.7, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D float16 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D float32 array with edge/special values
    x = np.array(
        [-np.inf, -1e10, -1.0, -0.0, 0.0, 1.0, 1e10, np.inf, np.nan],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D empty float32 array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D float64 array with near-integer boundaries
    x = np.array(
        [[-2.0001, -2.0, -1.9999], [1.9999, 2.0, 2.0001]], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 5D float32 array with singleton dimensions
    x = np.random.uniform(-5.0, 5.0, size=(1, 3, 1, 4, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D float64 array with large values
    x = np.array([1e20, -1e20, 1.5e15, -1.5e15], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.floor"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.floor'.")


check_valid('jax.lax.floor', generated_inputs['jax.lax.floor'], lib="jax", suffix=0)
