
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_max_inputs():
    list_of_inputs = []

    # Input 1: 0D boolean arrays (rank 0)
    x = np.array(True, dtype=bool)
    y = np.array(False, dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 1D boolean arrays (rank 1, same size)
    x = np.array([True, False, True, False, True], dtype=bool)
    y = np.array([False, False, True, True, False], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 2D boolean arrays (rank 2, same shape)
    x = np.random.choice([True, False], size=(2, 3))
    y = np.random.choice([True, False], size=(2, 3))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 3D boolean arrays (rank 3, same shape)
    x = np.random.choice([True, False], size=(2, 2, 2))
    y = np.random.choice([True, False], size=(2, 2, 2))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: 2D boolean arrays (rank 2, broadcasting 1x3 and 3x1)
    x = np.random.choice([True, False], size=(1, 3))
    y = np.random.choice([True, False], size=(3, 1))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: 2D boolean arrays (rank 2, broadcasting 1x5 and 2x5)
    x = np.random.choice([True, False], size=(1, 5))
    y = np.random.choice([True, False], size=(2, 5))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 3D boolean arrays (rank 3, broadcasting 1x2x1 and 2x1x2)
    x = np.random.choice([True, False], size=(1, 2, 1))
    y = np.random.choice([True, False], size=(2, 1, 2))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: 4D boolean arrays (rank 4, broadcasting 1x2x1x3 and 2x1x3x1)
    x = np.random.choice([True, False], size=(1, 2, 1, 3))
    y = np.random.choice([True, False], size=(2, 1, 3, 1))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Large 1D boolean arrays (rank 1)
    x = np.random.choice([True, False], size=(100,))
    y = np.random.choice([True, False], size=(100,))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 5D boolean arrays (rank 5, same shape)
    x = np.random.choice([True, False], size=(2, 2, 2, 2, 2))
    y = np.random.choice([True, False], size=(2, 2, 2, 2, 2))
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: 1D boolean arrays (rank 1, broadcasting 1 and 5)
    x = np.array([True], dtype=bool)
    y = np.array([False, True, False, True, False], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.lax.max_4"] = jax_lax_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.max_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.max_4'.")


check_valid('jax.lax.max', generated_inputs['jax.lax.max_4'], lib="jax", suffix=4)
