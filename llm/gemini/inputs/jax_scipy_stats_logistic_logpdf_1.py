
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logistic_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard logistic distribution
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with float32, matching shapes
    input_dict = {
        "x": np.array([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 1.0, -1.0, 2.0], dtype=np.float32),
        "scale": np.array([1.0, 0.5, 2.0, 1.5, 0.8], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array for x, 0D arrays for loc and scale (broadcasting)
    input_dict = {
        "x": np.linspace(-5.0, 5.0, 20, dtype=np.float64),
        "loc": np.array(0.0, dtype=np.float64),
        "scale": np.array(2.0, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, float64
    input_dict = {
        "x": np.array([[-1.0, 2.0], [0.5, -3.0]], dtype=np.float64),
        "loc": np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64),
        "scale": np.array([[0.5, 1.5], [2.0, 0.1]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D x with broadcasted 1D loc and scale
    input_dict = {
        "x": np.ones((3, 4), dtype=np.float32),
        "loc": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "scale": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays, float32
    input_dict = {
        "x": np.random.randn(2, 3, 4).astype(np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.ones((2, 3, 4), dtype=np.float32) * 1.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative loc and various scale values
    input_dict = {
        "x": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32),
        "loc": np.array([-5.0, -5.0, -5.0, -5.0, -5.0], dtype=np.float32),
        "scale": np.array([0.1, 0.5, 1.0, 5.0, 10.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays, float32
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "loc": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "scale": np.random.uniform(0.5, 3.0, size=(2, 2, 2, 2)).astype(
            np.float32
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values of x to test tails
    input_dict = {
        "x": np.array([100.0, -100.0, 50.0, -50.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed dimensional broadcasting (1D x, 2D loc, 0D scale)
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32),
        "scale": np.array(0.75, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.logistic.logpdf_1"] = logistic_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.logpdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.logpdf_1'.")


check_valid('jax.scipy.stats.logistic.logpdf', generated_inputs['jax.scipy.stats.logistic.logpdf_1'], lib="jax", suffix=1)
