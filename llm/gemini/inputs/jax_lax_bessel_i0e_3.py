
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bessel_i0e_inputs():
    list_of_inputs = []

    # Input 1: Positive integer-valued float32
    x = np.float32(5.0)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: Negative integer-valued float32
    x = np.float32(-10.0)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Zero float64
    x = np.float64(0.0)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Large positive integer-valued float64
    x = np.float64(100.0)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array of integer-valued float32
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array with mixed negative and positive integer-valued float32
    x = np.array([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array of integer-valued float64
    x = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 3D array of integer-valued float32
    x = np.arange(-12, 12, dtype=np.float32).reshape((2, 3, 4))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array of larger integer-valued float32
    x = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 4D array of integer-valued float32
    x = np.random.randint(-50, 50, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D array of type float16 (if supported, else float32)
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.bessel_i0e_3"] = bessel_i0e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i0e_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i0e_3'.")


check_valid('jax.lax.bessel_i0e', generated_inputs['jax.lax.bessel_i0e_3'], lib="jax", suffix=3)
