
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_r_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar arrays (floats)
    input_dict = {
        "x": np.array(1.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-D arrays, float32, positive values
    input_dict = {
        "x": np.array([0.5, 1.0, 2.5], dtype=np.float32),
        "loc": np.array([0.0, 0.5, 1.0], dtype=np.float32),
        "scale": np.array([1.0, 2.0, 0.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1-D arrays with negative values in x and loc, float64
    input_dict = {
        "x": np.array([-2.5, -0.5, 0.0, 3.2], dtype=np.float64),
        "loc": np.array([-1.0, -2.0, 0.0, 1.5], dtype=np.float64),
        "scale": np.array([0.5, 1.0, 2.0, 1.5], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2-D arrays, same shape
    input_dict = {
        "x": np.array([[-1.0, 2.0], [0.5, -3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 1.0], [-1.0, 0.5]], dtype=np.float32),
        "scale": np.array([[1.0, 0.5], [2.0, 1.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting - 2-D x, 1-D loc, scalar scale
    input_dict = {
        "x": np.ones((3, 4), dtype=np.float32),
        "loc": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "scale": np.array(2.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting - 3-D x with 1-D loc and 2-D scale
    input_dict = {
        "x": np.zeros((2, 3, 4), dtype=np.float64),
        "loc": np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float64),
        "scale": np.ones((3, 1), dtype=np.float64) * 1.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3-D arrays, all same shape
    input_dict = {
        "x": np.random.randn(2, 2, 3).astype(np.float32),
        "loc": np.random.randn(2, 2, 3).astype(np.float32),
        "scale": np.abs(np.random.randn(2, 2, 3)).astype(np.float32) + 0.1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4-D arrays
    input_dict = {
        "x": np.linspace(-3.0, 3.0, 16).reshape(2, 2, 2, 2).astype(np.float32),
        "loc": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2, 2), dtype=np.float32) * 0.8,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Extreme values for x (large positive and negative)
    input_dict = {
        "x": np.array([-50.0, -10.0, 0.0, 10.0, 50.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large scale values and varying loc
    input_dict = {
        "x": np.array([[100.0, -100.0], [50.0, -50.0]], dtype=np.float64),
        "loc": np.array([[10.0, -10.0], [5.0, -5.0]], dtype=np.float64),
        "scale": np.array([[50.0, 100.0], [25.0, 75.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_r.logcdf_1"] = (
    gumbel_r_logcdf_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.logcdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.logcdf_1'.")


check_valid('jax.scipy.stats.gumbel_r.logcdf', generated_inputs['jax.scipy.stats.gumbel_r.logcdf_1'], lib="jax", suffix=1)
