
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def digamma_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 scalar array (positive)
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D float32 array
    x = np.array([0.5, 1.0, 1.5, 2.0, 3.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float64 array with positive values
    x = np.array([[1.2, 2.3], [3.4, 4.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D float16 array
    x = np.array([0.1, 0.5, 1.0, 5.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float32 array with negative non-integer values
    x = np.array(
        [[[-0.5, -1.5], [-2.5, -3.5]], [[0.5, 1.5], [2.5, 3.5]]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D float32 array with random positive values
    x = np.abs(np.random.randn(2, 3, 4, 5).astype(np.float32)) + 0.1
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D float32 array with mixed positive and negative values
    x = np.array([[-0.2, 0.3, 1.4], [2.5, -3.1, 4.2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D float16 array
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D float64 array with large values
    x = np.array([1e2, 1e4, 1e6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 5D float32 array
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32) * 1.5
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D float64 array with small positive values close to zero
    x = np.array([1e-5, 1e-4, 1e-3, 1e-2], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.digamma_1"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.digamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.digamma_1'.")


check_valid('jax.lax.digamma', generated_inputs['jax.lax.digamma_1'], lib="jax", suffix=1)
