
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bessel_i1e_inputs():
    list_of_inputs = []

    # Input 1: positive float representing integer
    list_of_inputs.append({"x": np.float32(5.0)})

    # Input 2: zero float
    list_of_inputs.append({"x": np.float32(0.0)})

    # Input 3: negative float representing integer
    list_of_inputs.append({"x": np.float32(-5.0)})

    # Input 4: np.float64 scalar representing integer
    list_of_inputs.append({"x": np.float64(10.0)})

    # Input 5: np.float64 scalar representing negative integer
    list_of_inputs.append({"x": np.float64(-42.0)})

    # Input 6: 1D np.float32 array with integer values
    list_of_inputs.append({"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)})

    # Input 7: 1D np.float64 array with negative and zero integer values
    list_of_inputs.append({"x": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)})

    # Input 8: 2D np.float32 array with integer values
    list_of_inputs.append({"x": np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)})

    # Input 9: 3D np.float32 array with integer values
    list_of_inputs.append({"x": np.arange(-4, 4, dtype=np.float32).reshape(2, 2, 2)})

    # Input 10: 0-dimensional np.float32 array representing integer
    list_of_inputs.append({"x": np.array(7.0, dtype=np.float32)})

    # Input 11: large scalar np.float64 representing integer
    list_of_inputs.append({"x": np.float64(1000.0)})

    return list_of_inputs

generated_inputs["jax.lax.bessel_i1e_3"] = bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i1e_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i1e_3'.")


check_valid('jax.lax.bessel_i1e', generated_inputs['jax.lax.bessel_i1e_3'], lib="jax", suffix=3)
