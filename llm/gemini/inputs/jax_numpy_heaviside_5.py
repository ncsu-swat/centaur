
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: x1 is 0, x2 is 1D float32 array
    list_of_inputs.append({
        "x1": 0,
        "x2": np.array([0.5, 1.5, 2.5], dtype=np.float32)
    })

    # Input 2: x1 is negative, x2 is 2D float32 array
    list_of_inputs.append({
        "x1": -1,
        "x2": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    })

    # Input 3: x1 is positive, x2 is 1D float64 array
    list_of_inputs.append({
        "x1": 2,
        "x2": np.array([0.1, 0.2, 0.3], dtype=np.float64)
    })

    # Input 4: x1 is 0, x2 is 3D random float32 array
    list_of_inputs.append({
        "x1": 0,
        "x2": np.random.randn(2, 3, 2).astype(np.float32)
    })

    # Input 5: x1 is negative, x2 is 1D int32 array
    list_of_inputs.append({
        "x1": -5,
        "x2": np.ones((4,), dtype=np.int32)
    })

    # Input 6: x1 is 0, x2 is 2D float64 array
    list_of_inputs.append({
        "x1": 0,
        "x2": np.array([[-0.5, 0.5], [1.5, -1.5]], dtype=np.float64)
    })

    # Input 7: x1 is positive, x2 is 2D float32 array
    list_of_inputs.append({
        "x1": 10,
        "x2": np.random.rand(3, 4).astype(np.float32)
    })

    # Input 8: x1 is 0, x2 is 1D int64 array
    list_of_inputs.append({
        "x1": 0,
        "x2": np.array([1, 2, 3, 4, 5], dtype=np.int64)
    })

    # Input 9: x1 is negative, x2 is 3D float32 array
    list_of_inputs.append({
        "x1": -100,
        "x2": np.zeros((2, 2, 2), dtype=np.float32)
    })

    # Input 10: x1 is positive, x2 is 4D float32 array
    list_of_inputs.append({
        "x1": 1,
        "x2": np.random.randn(1, 2, 1, 3).astype(np.float32)
    })

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_5"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_5'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_5'], lib="jax", suffix=5)
