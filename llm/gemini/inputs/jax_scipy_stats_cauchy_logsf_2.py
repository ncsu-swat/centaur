
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_logsf_inputs():
    list_of_inputs = []

    # Input 1: Standard Cauchy, 1D array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-standard 1D array, positive loc, scale > 1
    x = np.array([-10.0, 0.0, 10.0, 20.0], dtype=np.float32)
    input_dict = {"x": x, "loc": 5.0, "scale": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float64, negative loc, small scale
    x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    input_dict = {"x": x, "loc": -1.0, "scale": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, large scale
    x = np.random.uniform(-50, 50, (2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "loc": 10.0, "scale": 20.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with float64, extreme values
    x = np.array([-1000.0, -100.0, 100.0, 1000.0], dtype=np.float64)
    input_dict = {"x": x, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, standard parameters
    x = np.random.randn(2, 3, 2, 2).astype(np.float32)
    input_dict = {"x": x, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, very small scale
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    input_dict = {"x": x, "loc": 0.0, "scale": 0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, negative loc, scale > 1
    x = np.random.uniform(-5, 5, (3, 3)).astype(np.float32)
    input_dict = {"x": x, "loc": -3.5, "scale": 4.2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array (high dimension), float32
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {"x": x, "loc": 1.5, "scale": 0.8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, loc and scale are floats
    x = np.array([-5.0, -2.5, 2.5, 5.0], dtype=np.float32)
    input_dict = {"x": x, "loc": -1.2, "scale": 3.14}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.logsf_2"] = jax_scipy_stats_cauchy_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logsf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logsf_2'.")


check_valid('jax.scipy.stats.cauchy.logsf', generated_inputs['jax.scipy.stats.cauchy.logsf_2'], lib="jax", suffix=2)
