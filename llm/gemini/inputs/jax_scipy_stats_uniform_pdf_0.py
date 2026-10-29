
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def uniform_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, values inside the uniform interval
    input_dict = {
        "x": np.array([0.1, 0.5, 0.9], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays with values outside the interval
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting scalar loc and scale over 2D x
    input_dict = {
        "x": np.linspace(-2.0, 5.0, 12, dtype=np.float32).reshape(3, 4),
        "loc": np.array(1.0, dtype=np.float32),
        "scale": np.array(3.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative loc values and custom scale
    input_dict = {
        "x": np.array([-5.0, -4.0, -2.5, 0.0], dtype=np.float32),
        "loc": np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32),
        "scale": np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays with matching shapes
    input_dict = {
        "x": np.ones((2, 3), dtype=np.float64) * 2.0,
        "loc": np.zeros((2, 3), dtype=np.float64),
        "scale": np.ones((2, 3), dtype=np.float64) * 5.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting 1D parameters across 2D x
    input_dict = {
        "x": np.zeros((4, 3), dtype=np.float32),
        "loc": np.array([0.0, -1.0, 2.0], dtype=np.float32),
        "scale": np.array([2.0, 4.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensors
    input_dict = {
        "x": np.random.uniform(0.0, 10.0, size=(2, 3, 4)).astype(np.float32),
        "loc": np.full((2, 3, 4), 2.0, dtype=np.float32),
        "scale": np.full((2, 3, 4), 6.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large scale values and float64 precision
    input_dict = {
        "x": np.array([500.0, 1000.0, 1500.0], dtype=np.float64),
        "loc": np.array([100.0, 100.0, 100.0], dtype=np.float64),
        "scale": np.array([1000.0, 1000.0, 1000.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensor inputs
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "loc": np.zeros((1, 2, 1, 2), dtype=np.float32),
        "scale": np.ones((2, 1, 2, 1), dtype=np.float32) * 2.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Non-standard float scale < 1.0
    input_dict = {
        "x": np.array([0.01, 0.05, 0.09], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.1, 0.1, 0.1], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.uniform.pdf"] = uniform_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.uniform.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.uniform.pdf'.")


check_valid('jax.scipy.stats.uniform.pdf', generated_inputs['jax.scipy.stats.uniform.pdf'], lib="jax", suffix=0)
