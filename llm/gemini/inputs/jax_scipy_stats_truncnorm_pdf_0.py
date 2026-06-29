
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def truncnorm_pdf_inputs():
    list_of_inputs = []

    # Input 1, valid — 0D scalars, float32, standard normal bounds
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(-1.0, dtype=np.float32),
        "b": np.array(1.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid — 1D arrays, float32, length 5
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "a": np.array([-2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid — float64, 1D arrays, broadcasting 1D x with 0D params
    input_dict = {
        "x": np.array([-0.5, 0.0, 0.5], dtype=np.float64),
        "a": np.array(-1.5, dtype=np.float64),
        "b": np.array(1.5, dtype=np.float64),
        "loc": np.array(0.1, dtype=np.float64),
        "scale": np.array(0.8, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid — 2D arrays, shape (2, 3)
    input_dict = {
        "x": np.array([[0.0, 0.5, 1.0], [-1.0, -0.5, 0.0]], dtype=np.float32),
        "a": np.array([[-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0]], dtype=np.float32),
        "b": np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.5, 2.0], [1.0, 1.5, 2.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid — bounds in positive range, elements outside and inside interval
    input_dict = {
        "x": np.array([0.0, 1.5, 4.0], dtype=np.float32),
        "a": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "b": np.array([3.0, 3.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid — bounds in negative range
    input_dict = {
        "x": np.array([-4.0, -2.0, 0.0], dtype=np.float32),
        "a": np.array([-3.0, -3.0, -3.0], dtype=np.float32),
        "b": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid — complex broadcasting with shapes (3, 1) and (1, 3)
    input_dict = {
        "x": np.array([[0.0], [0.5], [1.0]], dtype=np.float32),
        "a": np.array([[-1.0, -2.0, -3.0]], dtype=np.float32),
        "b": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "loc": np.array([[0.0]], dtype=np.float32),
        "scale": np.array([[1.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid — 3D arrays, float32
    input_dict = {
        "x": np.zeros((2, 2, 2), dtype=np.float32),
        "a": np.ones((2, 2, 2), dtype=np.float32) * -2.0,
        "b": np.ones((2, 2, 2), dtype=np.float32) * 2.0,
        "loc": np.ones((2, 2, 2), dtype=np.float32) * 0.5,
        "scale": np.ones((2, 2, 2), dtype=np.float32) * 1.2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid — small scale parameter, float64
    input_dict = {
        "x": np.array([0.1, 0.2], dtype=np.float64),
        "a": np.array([-0.5, -0.5], dtype=np.float64),
        "b": np.array([0.5, 0.5], dtype=np.float64),
        "loc": np.array([0.0, 0.0], dtype=np.float64),
        "scale": np.array([0.1, 0.1], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid — large scale parameter, float32
    input_dict = {
        "x": np.array([-10.0, 10.0], dtype=np.float32),
        "a": np.array([-5.0, -5.0], dtype=np.float32),
        "b": np.array([5.0, 5.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([10.0, 10.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.truncnorm.pdf"] = truncnorm_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.pdf'.")


check_valid('jax.scipy.stats.truncnorm.pdf', generated_inputs['jax.scipy.stats.truncnorm.pdf'], lib="jax", suffix=0)
