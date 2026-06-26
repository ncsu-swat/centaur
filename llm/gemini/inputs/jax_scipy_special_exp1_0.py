
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def exp1_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats (float32)
    x = np.array([0.1, 1.0, 2.5, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array of positive floats (float32)
    x = np.array([[0.5, 1.5], [2.0, 3.5]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D array of positive floats (float32)
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D array of positive double precision floats (float64)
    x = np.array([0.01, 0.5, 1.2, 100.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D array (scalar) (float32)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Large positive values (float32)
    x = np.array([50.0, 100.0, 200.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Small positive values (float32)
    x = np.array([1e-5, 1e-4, 1e-3, 1e-2], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D array with float64
    x = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D array of positive floats (float32)
    x = np.random.uniform(0.5, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array containing a wider range of positive values (float32)
    x = np.array([0.2, 0.8, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.scipy.special.exp1"] = exp1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.exp1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.exp1'.")


check_valid('jax.scipy.special.exp1', generated_inputs['jax.scipy.special.exp1'], lib="jax", suffix=0)
