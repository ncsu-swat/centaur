
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gamma_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard float32
    input_dict = {
        "x": np.array(1.5, dtype=np.float32),
        "a": np.array(2.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with float64
    input_dict = {
        "x": np.array([0.5, 1.0, 2.5, 5.0], dtype=np.float64),
        "a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting scalar loc and scale with 1D x and a
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "a": np.array([2.5, 2.5, 2.5], dtype=np.float32),
        "loc": np.array(0.5, dtype=np.float32),
        "scale": np.array(2.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, matching shapes
    input_dict = {
        "x": np.array([[0.1, 1.2], [2.3, 3.4]], dtype=np.float32),
        "a": np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        "loc": np.array([[0.0, 0.5], [-1.0, 1.0]], dtype=np.float32),
        "scale": np.array([[1.0, 0.5], [2.0, 1.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x < loc (testing support boundaries where pdf is 0)
    input_dict = {
        "x": np.array([-1.0, 0.0, 0.5, 2.0], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array broadcasting with shape (2, 1, 3) and (2, 4, 1)
    input_dict = {
        "x": np.ones((2, 4, 3), dtype=np.float32) * 3.0,
        "a": np.full((2, 1, 3), 1.5, dtype=np.float32),
        "loc": np.zeros((1, 4, 1), dtype=np.float32),
        "scale": np.ones((1, 1, 1), dtype=np.float32) * 0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small shape parameter (a < 1)
    input_dict = {
        "x": np.array([0.01, 0.05, 0.1, 0.5], dtype=np.float64),
        "a": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values of a and x
    input_dict = {
        "x": np.array([50.0, 100.0, 150.0], dtype=np.float64),
        "a": np.array([50.0, 100.0, 150.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative loc parameter
    input_dict = {
        "x": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "a": np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32),
        "loc": np.array([-3.0, -3.0, -3.0, -3.0, -3.0], dtype=np.float32),
        "scale": np.array([2.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with column broadcasting
    input_dict = {
        "x": np.linspace(0.5, 5.0, 12, dtype=np.float32).reshape(4, 3),
        "a": np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        "loc": np.array([0.0, 0.1, 0.2], dtype=np.float32),
        "scale": np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gamma.pdf"] = gamma_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.pdf'.")


check_valid('jax.scipy.stats.gamma.pdf', generated_inputs['jax.scipy.stats.gamma.pdf'], lib="jax", suffix=0)
