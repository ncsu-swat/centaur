
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def divmod_inputs():
    list_of_inputs = []

    # Input 1: positive float and 1D float32 tensor
    list_of_inputs.append({
        "x1": 10.0,
        "x2": np.array([3.0, 4.0, 5.0], dtype=np.float32)
    })

    # Input 2: negative float and 1D float32 tensor
    list_of_inputs.append({
        "x1": -10.0,
        "x2": np.array([3.0, 4.0, 5.0], dtype=np.float32)
    })

    # Input 3: positive float and 2D float32 tensor
    list_of_inputs.append({
        "x1": 5.5,
        "x2": np.array([[1.5, 2.0], [2.5, 3.0]], dtype=np.float32)
    })

    # Input 4: negative float and 2D float64 tensor
    list_of_inputs.append({
        "x1": -12.3,
        "x2": np.array([[2.1, -3.4], [-4.5, 5.6]], dtype=np.float64)
    })

    # Input 5: positive float and 3D float32 tensor
    list_of_inputs.append({
        "x1": 100.0,
        "x2": np.ones((2, 2, 2), dtype=np.float32) * 3.0
    })

    # Input 6: float and 1D int32 tensor
    list_of_inputs.append({
        "x1": 7.0,
        "x2": np.array([2, 3, 4], dtype=np.int32)
    })

    # Input 7: float and 1D int64 tensor with negative values
    list_of_inputs.append({
        "x1": -8.5,
        "x2": np.array([-3, -2, 2, 3], dtype=np.int64)
    })

    # Input 8: float and 0D float32 tensor
    list_of_inputs.append({
        "x1": 9.0,
        "x2": np.array(2.5, dtype=np.float32)
    })

    # Input 9: float and 2D float32 tensor with random values
    list_of_inputs.append({
        "x1": 15.75,
        "x2": np.random.uniform(1.0, 10.0, (5, 5)).astype(np.float32)
    })

    # Input 10: float and 4D float64 tensor
    list_of_inputs.append({
        "x1": 1.0,
        "x2": np.ones((2, 2, 2, 2), dtype=np.float64) * 0.3
    })

    # Input 11: float and 1D float16 tensor
    list_of_inputs.append({
        "x1": 5.0,
        "x2": np.array([1.5, 2.5], dtype=np.float16)
    })

    return list_of_inputs

generated_inputs["jax.numpy.divmod_5"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_5'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_5'], lib="jax", suffix=5)
