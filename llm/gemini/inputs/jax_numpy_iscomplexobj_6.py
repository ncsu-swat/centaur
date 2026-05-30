
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: tuple of real integers
    list_of_inputs.append({"x": (1, 2, 3)})

    # Input 2: tuple of real floats
    list_of_inputs.append({"x": (1.0, -2.5, 3.14)})

    # Input 3: tuple with a complex number
    list_of_inputs.append({"x": (1, 2.0, 3 + 4j)})

    # Input 4: tuple of complex numbers
    list_of_inputs.append({"x": (1j, 2 - 3j)})

    # Input 5: tuple containing numpy array of float32
    list_of_inputs.append({"x": (np.array([1.0, 2.0], dtype=np.float32),)})

    # Input 6: tuple containing numpy array of complex64
    list_of_inputs.append({"x": (np.array([1.0 + 1j, 2.0], dtype=np.complex64),)})

    # Input 7: tuple containing mix of bool and complex
    list_of_inputs.append({"x": (True, 1j)})

    # Input 8: tuple with multi-dimensional numpy array of complex
    list_of_inputs.append({"x": (np.array([[1j, 2j], [3j, 4j]], dtype=np.complex128),)})

    # Input 9: empty tuple
    list_of_inputs.append({"x": ()})

    # Input 10: tuple containing numpy complex scalar and real scalar
    list_of_inputs.append({"x": (np.array(1 + 2j, dtype=np.complex64), np.array(3.0, dtype=np.float64))})

    # Input 11: tuple of negative complex numbers
    list_of_inputs.append({"x": (-1j, -2 - 3j, -4.5j)})

    return list_of_inputs

generated_inputs["jax.numpy.iscomplexobj_6"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_6'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_6'], lib="jax", suffix=6)
