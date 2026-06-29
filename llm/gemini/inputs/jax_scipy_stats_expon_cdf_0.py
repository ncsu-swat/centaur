
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expon_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, standard values (float32)
    input_dict = {
        "x": np.array(1.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, same size (float32)
    input_dict = {
        "x": np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32),
        "loc": np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32),
        "scale": np.array([1.0, 1.5, 2.0, 0.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, same size (float64)
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "loc": np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64),
        "scale": np.array([[1.0, 2.0], [1.5, 0.5]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting scalar loc and scale to 1D x
    input_dict = {
        "x": np.array([0.1, 0.5, 1.0, 10.0], dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(2.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting 1D loc/scale to 2D x
    input_dict = {
        "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "loc": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "scale": np.array([0.5, 1.0, 1.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative x values and x < loc
    input_dict = {
        "x": np.array([-1.0, -0.5, 0.0, 1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large scale values
    input_dict = {
        "x": np.array([10.0, 100.0, 1000.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([100.0, 500.0, 1000.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small scale values (but positive)
    input_dict = {
        "x": np.array([0.01, 0.1, 1.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([0.001, 0.01, 0.1], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays
    input_dict = {
        "x": np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32),
        "loc": np.random.uniform(0.0, 0.5, size=(2, 2, 2)).astype(np.float32),
        "scale": np.random.uniform(0.5, 2.0, size=(2, 2, 2)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting with dimensions (3, 1) and (1, 3)
    input_dict = {
        "x": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.5, 2.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.expon.cdf"] = expon_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.cdf'.")


check_valid('jax.scipy.stats.expon.cdf', generated_inputs['jax.scipy.stats.expon.cdf'], lib="jax", suffix=0)
