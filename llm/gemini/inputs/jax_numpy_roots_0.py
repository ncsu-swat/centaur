
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def roots_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 array, strip_zeros=True
    p = np.array([1.0, -2.0, 1.0], dtype=np.float32)
    strip_zeros = True
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 2: Float32 array with leading zeros, strip_zeros=True
    p = np.array([0.0, 0.0, 1.0, -5.0, 6.0], dtype=np.float32)
    strip_zeros = True
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 3: Float32 array with leading zeros, strip_zeros=False
    p = np.array([0.0, 0.0, 1.0, -5.0, 6.0], dtype=np.float32)
    strip_zeros = False
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 4: Float64 array with negative and positive values
    p = np.array([-3.5, 2.0, -1.5, 10.0], dtype=np.float64)
    strip_zeros = True
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 5: Integer array
    p = np.array([1, -3, 2], dtype=np.int32)
    strip_zeros = True
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 6: Complex64 array
    p = np.array([1.0 + 1.0j, 2.0 - 3.0j, -5.0j], dtype=np.complex64)
    strip_zeros = True
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 7: Complex128 array, strip_zeros=False
    p = np.array([0.0 + 0.0j, 1.0 + 2.0j, -3.0j], dtype=np.complex128)
    strip_zeros = False
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 8: High degree polynomial, float32
    p = np.array([1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    strip_zeros = True
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 9: Single element array (constant polynomial)
    p = np.array([5.0], dtype=np.float32)
    strip_zeros = True
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 10: Array with only zeros, strip_zeros=False
    p = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    strip_zeros = False
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    # Input 11: All elements negative, float64
    p = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64)
    strip_zeros = True
    list_of_inputs.append({"p": p, "strip_zeros": strip_zeros})

    return list_of_inputs

generated_inputs["jax.numpy.roots"] = roots_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.roots' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.roots'.")


check_valid('jax.numpy.roots', generated_inputs['jax.numpy.roots'], lib="jax", suffix=0)
