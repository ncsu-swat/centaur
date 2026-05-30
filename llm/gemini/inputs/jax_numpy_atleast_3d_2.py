
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atleast_3d_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array (representing two 1D arrays if unpacked)
    list_of_inputs.append({
        "arys": np.random.randn(2, 3).astype(np.float32)
    })

    # Input 2: 3D float64 array (representing three 2D arrays if unpacked)
    list_of_inputs.append({
        "arys": np.random.randn(3, 4, 5).astype(np.float64)
    })

    # Input 3: 1D int32 array (representing five scalars if unpacked)
    list_of_inputs.append({
        "arys": np.arange(5).astype(np.int32)
    })

    # Input 4: 4D int64 array (representing two 3D arrays if unpacked)
    list_of_inputs.append({
        "arys": np.ones((2, 2, 2, 2), dtype=np.int64)
    })

    # Input 5: 2D float16 array (representing one 1D array if unpacked)
    list_of_inputs.append({
        "arys": np.zeros((1, 5), dtype=np.float16)
    })

    # Input 6: 2D boolean array
    list_of_inputs.append({
        "arys": np.array([[True, False], [False, True]])
    })

    # Input 7: 3D float32 array with negative values
    list_of_inputs.append({
        "arys": (np.random.randn(4, 2, 3) * 10).astype(np.float32)
    })

    # Input 8: 2D int32 array
    list_of_inputs.append({
        "arys": np.arange(12).reshape(2, 6).astype(np.int32)
    })

    # Input 9: 3D float32 array with single-dimensions
    list_of_inputs.append({
        "arys": np.ones((3, 1, 1), dtype=np.float32)
    })

    # Input 10: 1D float32 array
    list_of_inputs.append({
        "arys": np.array([1.5, -2.5, 3.5], dtype=np.float32)
    })

    return list_of_inputs

generated_inputs["jax.numpy.atleast_3d_2"] = atleast_3d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_3d_2'.")


check_valid('jax.numpy.atleast_3d', generated_inputs['jax.numpy.atleast_3d_2'], lib="jax", suffix=2)
