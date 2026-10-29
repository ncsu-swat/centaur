
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float32, standard values
    input_dict = {
        "x": np.array([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float64, various loc and scale
    input_dict = {
        "x": np.array([[-1.0, 2.0], [0.5, -3.0]], dtype=np.float64),
        "loc": np.array([[0.0, 1.0], [-1.0, 0.5]], dtype=np.float64),
        "scale": np.array([[0.5, 2.0], [1.5, 0.8]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting with scalar-like 1D scale/loc
    input_dict = {
        "x": np.array([[-1.0, 0.0, 1.0], [2.0, -0.5, 0.2]], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([2.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensors, float32
    input_dict = {
        "x": np.zeros((2, 3, 4), dtype=np.float32),
        "loc": np.ones((2, 3, 4), dtype=np.float32) * -0.5,
        "scale": np.ones((2, 3, 4), dtype=np.float32) * 1.2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large negative x values
    input_dict = {
        "x": np.array([-10.0, -20.0, -50.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large positive x values
    input_dict = {
        "x": np.array([10.0, 20.0, 50.0], dtype=np.float64),
        "loc": np.array([5.0, 5.0, 5.0], dtype=np.float64),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting shapes (3, 1) and (1, 4)
    input_dict = {
        "x": np.array([[-1.0], [0.0], [1.0]], dtype=np.float32),
        "loc": np.array([[0.0, 1.0, -1.0, 0.5]], dtype=np.float32),
        "scale": np.array([[1.0, 0.5, 2.0, 1.5]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensors, float32
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "loc": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "scale": np.abs(np.random.randn(2, 2, 2, 2)).astype(np.float32) + 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scale significantly smaller than 1
    input_dict = {
        "x": np.array([0.01, 0.02, 0.03], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([0.01, 0.05, 0.1], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Scale significantly larger than 1
    input_dict = {
        "x": np.array([100.0, -50.0, 0.0], dtype=np.float32),
        "loc": np.array([50.0, 0.0, -25.0], dtype=np.float32),
        "scale": np.array([100.0, 200.0, 50.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.logcdf_1"] = gumbel_l_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.logcdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.logcdf_1'.")


check_valid('jax.scipy.stats.gumbel_l.logcdf', generated_inputs['jax.scipy.stats.gumbel_l.logcdf_1'], lib="jax", suffix=1)
