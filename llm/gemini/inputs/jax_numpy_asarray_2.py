
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def asarray_inputs():
    list_of_inputs = []

    # Input 1: 1D list of floats, float32, copy True, order 'K'
    list_of_inputs.append({
        "a": [1.0, 2.5, -3.2, 4.7],
        "dtype": np.dtype("float32"),
        "order": "K",
        "copy": True
    })

    # Input 2: 1D list of ints, int32, copy False, order 'K'
    list_of_inputs.append({
        "a": [-10, 0, 10, 20],
        "dtype": np.dtype("int32"),
        "order": "K",
        "copy": False
    })

    # Input 3: 2D list of floats, float64, copy True, order 'K'
    list_of_inputs.append({
        "a": [[1.0, 2.0], [3.0, 4.0]],
        "dtype": np.dtype("float64"),
        "order": "K",
        "copy": True
    })

    # Input 4: 2D list of ints, int16, copy False, order 'K'
    list_of_inputs.append({
        "a": [[1, 2, 3], [4, 5, 6]],
        "dtype": np.dtype("int16"),
        "order": "K",
        "copy": False
    })

    # Input 5: 1D list of bools, bool, copy True, order 'K'
    list_of_inputs.append({
        "a": [True, False, True, True],
        "dtype": np.dtype("bool"),
        "order": "K",
        "copy": True
    })

    # Input 6: 3D nested list of bools, bool, copy False, order 'K'
    list_of_inputs.append({
        "a": [[[True, False], [False, True]], [[True, True], [False, False]]],
        "dtype": np.dtype("bool"),
        "order": "K",
        "copy": False
    })

    # Input 7: 1D list of complex numbers, complex64, copy True, order 'K'
    list_of_inputs.append({
        "a": [1 + 1j, 2 - 3j, 0.5j],
        "dtype": np.dtype("complex64"),
        "order": "K",
        "copy": True
    })

    # Input 8: 3D list of floats, float32, copy False, order 'K'
    list_of_inputs.append({
        "a": [[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]],
        "dtype": np.dtype("float32"),
        "order": "K",
        "copy": False
    })

    # Input 9: Empty list, float32, copy True, order 'K'
    list_of_inputs.append({
        "a": [],
        "dtype": np.dtype("float32"),
        "order": "K",
        "copy": True
    })

    # Input 10: 2D mixed numerical list, float64, copy True, order 'K'
    list_of_inputs.append({
        "a": [[1, 2.5], [3, 4.2]],
        "dtype": np.dtype("float64"),
        "order": "K",
        "copy": True
    })

    return list_of_inputs

generated_inputs["jax.numpy.asarray_2"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asarray_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asarray_2'.")


check_valid('jax.numpy.asarray', generated_inputs['jax.numpy.asarray_2'], lib="jax", suffix=2)
