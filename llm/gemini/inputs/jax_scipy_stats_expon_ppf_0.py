
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expon_ppf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard exponential
    input_dict = {
        "q": np.array(0.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with float64
    input_dict = {
        "q": np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float64),
        "loc": np.array([0.0, 1.0, -1.0, 2.0, 0.5], dtype=np.float64),
        "scale": np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays with broadcasting
    input_dict = {
        "q": np.linspace(0.01, 0.99, 12, dtype=np.float32).reshape(3, 4),
        "loc": np.array([[-2.0], [0.0], [3.0]], dtype=np.float32),
        "scale": np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array with identical shapes
    shape = (2, 3, 4)
    input_dict = {
        "q": np.random.uniform(0.0, 1.0, size=shape).astype(np.float32),
        "loc": np.random.uniform(-5.0, 5.0, size=shape).astype(np.float32),
        "scale": np.random.uniform(0.1, 10.0, size=shape).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boundary probabilities q=0 and q=1
    input_dict = {
        "q": np.array([0.0, 1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large scale values
    input_dict = {
        "q": np.array([0.05, 0.5, 0.95], dtype=np.float64),
        "loc": np.array([100.0, 200.0, -50.0], dtype=np.float64),
        "scale": np.array([1000.0, 500.0, 2500.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small scale values (steep rate)
    input_dict = {
        "q": np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32),
        "loc": np.zeros((2, 2), dtype=np.float32),
        "scale": np.array([[1e-3, 1e-4], [1e-2, 1e-5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative locations
    input_dict = {
        "q": np.array([0.3, 0.6, 0.9], dtype=np.float32),
        "loc": np.array([-10.0, -20.0, -30.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher-dimensional broadcast (4D and 1D)
    input_dict = {
        "q": np.random.uniform(0.1, 0.9, size=(2, 1, 3, 4)).astype(np.float64),
        "loc": np.array([1.5], dtype=np.float64),
        "scale": np.random.uniform(0.5, 2.5, size=(4,)).astype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty arrays
    input_dict = {
        "q": np.array([], dtype=np.float32),
        "loc": np.array([], dtype=np.float32),
        "scale": np.array([], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Probabilities close to extreme boundaries
    input_dict = {
        "q": np.array([1e-7, 0.5, 1.0 - 1e-7], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.expon.ppf"] = expon_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.ppf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.ppf'.")


check_valid('jax.scipy.stats.expon.ppf', generated_inputs['jax.scipy.stats.expon.ppf'], lib="jax", suffix=0)
