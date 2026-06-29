
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_le_inputs():
    list_of_inputs = []

    # Input 1: Scalar-like 0D arrays
    x = np.array(True, dtype=np.bool_)
    y = np.array(False, dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 1D boolean arrays of same size
    x = np.array([True, False, True], dtype=np.bool_)
    y = np.array([False, True, True], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 2D boolean arrays of same size
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([[False, True], [True, False]], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Scalar 0D array and 1D array
    x = np.array(False, dtype=np.bool_)
    y = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Scalar 0D array and 2D array
    x = np.array(True, dtype=np.bool_)
    y = np.array([[False, True], [True, False]], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 3D boolean arrays of same shape
    x = np.ones((2, 3, 2), dtype=np.bool_)
    y = np.zeros((2, 3, 2), dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Broadcasting same number of dimensions (3D)
    x = np.array([[[True, False]], [[False, True]]], dtype=np.bool_)  # shape (2, 1, 2)
    y = np.ones((2, 3, 2), dtype=np.bool_)  # shape (2, 3, 2)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Random 1D booleans of same shape
    x = np.random.choice([True, False], size=10).astype(np.bool_)
    y = np.random.choice([True, False], size=10).astype(np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Random 2D booleans of same shape
    x = np.random.choice([True, False], size=(3, 3)).astype(np.bool_)
    y = np.random.choice([True, False], size=(3, 3)).astype(np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 4D boolean arrays of same shape
    x = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(np.bool_)
    y = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: 3D boolean array and 0D array scalar
    x = np.ones((2, 2, 2), dtype=np.bool_)
    y = np.array(True, dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.lax.le_4"] = jax_lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_4'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_4'], lib="jax", suffix=4)
