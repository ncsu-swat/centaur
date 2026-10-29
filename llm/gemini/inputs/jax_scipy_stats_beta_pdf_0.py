
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def beta_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard beta support [0, 1]
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(5.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with float64
    input_dict = {
        "x": np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float64),
        "a": np.array([1.5, 2.0, 3.0, 2.5], dtype=np.float64),
        "b": np.array([2.5, 3.0, 2.0, 1.5], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays with uniform shape parameters (a=1, b=1)
    input_dict = {
        "x": np.array([[0.2, 0.5], [0.8, 0.3]], dtype=np.float32),
        "a": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "b": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Non-zero loc and non-unit scale, values scaled to range [-2, 3] (loc=-2, scale=5)
    input_dict = {
        "x": np.array([-1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "b": np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32),
        "loc": np.array([-2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float32),
        "scale": np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasted shapes (x 2D, a 1D, b scalar, loc/scale scalars)
    input_dict = {
        "x": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "a": np.array([0.5, 1.0, 2.0], dtype=np.float32),
        "b": np.array(0.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Values outside standard support [0, 1] (checking evaluation bounds)
    input_dict = {
        "x": np.array([-0.5, 0.2, 0.8, 1.5], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "b": np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor inputs
    input_dict = {
        "x": np.full((2, 2, 2), 0.6, dtype=np.float64),
        "a": np.full((2, 2, 2), 3.5, dtype=np.float64),
        "b": np.full((2, 2, 2), 1.5, dtype=np.float64),
        "loc": np.full((2, 2, 2), 0.1, dtype=np.float64),
        "scale": np.full((2, 2, 2), 0.8, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boundary points x = loc and x = loc + scale with a > 1, b > 1
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small fractional shape parameters
    input_dict = {
        "x": np.array([0.25, 0.5, 0.75], dtype=np.float32),
        "a": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "b": np.array([0.3, 0.2, 0.1], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Column and row broadcasting (2x1 and 1x3)
    input_dict = {
        "x": np.array([[0.3], [0.7]], dtype=np.float32),
        "a": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "b": np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
        "loc": np.array([[0.0]], dtype=np.float32),
        "scale": np.array([[1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative location parameter
    input_dict = {
        "x": np.array([-9.0, -7.0, -5.0, -3.0, -1.0], dtype=np.float64),
        "a": np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64),
        "b": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        "loc": np.array([-10.0, -10.0, -10.0, -10.0, -10.0], dtype=np.float64),
        "scale": np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.beta.pdf"] = beta_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.pdf'.")


check_valid('jax.scipy.stats.beta.pdf', generated_inputs['jax.scipy.stats.beta.pdf'], lib="jax", suffix=0)
