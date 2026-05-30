
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_modf_inputs():
    list_of_inputs = []

    # Input 1: Basic positive and negative floats
    input_dict = {
        "x": np.array([1.5, 2.7, -3.2], dtype=np.float32),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Values near zero
    input_dict = {
        "x": np.array([-0.1, -0.9, 0.0, 0.5], dtype=np.float64),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger float values
    input_dict = {
        "x": np.array([100.25, -200.75], dtype=np.float32),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Special floating point values (nan, inf)
    input_dict = {
        "x": np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integers
    input_dict = {
        "x": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small values and larger precision
    input_dict = {
        "x": np.array([1e-6, -1e-6, 12345.6789], dtype=np.float64),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element array
    input_dict = {
        "x": np.array([0.0], dtype=np.float32),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mix of negative and positive floats
    input_dict = {
        "x": np.array([-3.4, -5.7, 0.6, 1.5, 2.3], dtype=np.float32),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float16 dtype compatible inputs
    input_dict = {
        "x": np.array([10.5, 20.5, 30.5, 40.5, 50.5, 60.5], dtype=np.float16),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High precision small differences
    input_dict = {
        "x": np.array([-0.0001, 0.0001], dtype=np.float32),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.modf_5"] = jax_numpy_modf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.modf_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.modf_5'.")


check_valid('jax.numpy.modf', generated_inputs['jax.numpy.modf_5'], lib="jax", suffix=5)
