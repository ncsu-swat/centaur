
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_eq_inputs():
    list_of_inputs = []

    # Input 1: Scalar boolean arrays
    x = np.array(True, dtype=bool)
    y = np.array(False, dtype=bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 1D boolean arrays of same shape
    x = np.array([True, False, True], dtype=bool)
    y = np.array([False, False, True], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 2D boolean arrays of same shape
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = np.array([[True, True], [False, False]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Broadcasting with same number of dimensions (2D to 2D)
    x = np.array([[True, False]], dtype=bool)
    y = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Broadcasting scalar to 2D array
    x = np.array(True, dtype=bool)
    y = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: 3D random boolean arrays
    x = np.random.choice([True, False], size=(2, 3, 2)).astype(bool)
    y = np.random.choice([True, False], size=(2, 3, 2)).astype(bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Broadcasting compatible shapes (1, 3) and (3, 1) - both 2D
    x = np.array([[True, False, True]], dtype=bool)
    y = np.array([[True], [False], [True]], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: 4D random boolean arrays
    x = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(bool)
    y = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Single element 1D arrays
    x = np.array([True], dtype=bool)
    y = np.array([False], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Larger 1D boolean arrays
    x = np.random.choice([True, False], size=(50,)).astype(bool)
    y = np.random.choice([True, False], size=(50,)).astype(bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.lax.eq_4"] = jax_lax_eq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.eq_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.eq_4'.")


check_valid('jax.lax.eq', generated_inputs['jax.lax.eq_4'], lib="jax", suffix=4)
