
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bessel_i1e_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar representing True
    list_of_inputs.append({"x": np.float32(1.0)})

    # Input 2: float32 scalar representing False
    list_of_inputs.append({"x": np.float32(0.0)})

    # Input 3: 1D float32 array representing [True]
    list_of_inputs.append({"x": np.array([1.0], dtype=np.float32)})

    # Input 4: 1D float32 array representing [True, False, True]
    list_of_inputs.append({"x": np.array([1.0, 0.0, 1.0], dtype=np.float32)})

    # Input 5: 2D float32 array representing [[True, False], [False, True]]
    list_of_inputs.append({"x": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)})

    # Input 6: 2D float32 array representing all True
    list_of_inputs.append({"x": np.ones((3, 3), dtype=np.float32)})

    # Input 7: 2D float32 array representing all False
    list_of_inputs.append({"x": np.zeros((2, 5), dtype=np.float32)})

    # Input 8: 3D float32 array representing booleans
    list_of_inputs.append({"x": np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)})

    # Input 9: High-dimensional float32 array representing all True
    list_of_inputs.append({"x": np.ones((2, 2, 2, 2), dtype=np.float32)})

    # Input 10: 2D float64 array representing booleans
    list_of_inputs.append({"x": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)})

    return list_of_inputs

generated_inputs["jax.lax.bessel_i1e_4"] = bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i1e_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i1e_4'.")


check_valid('jax.lax.bessel_i1e', generated_inputs['jax.lax.bessel_i1e_4'], lib="jax", suffix=4)
