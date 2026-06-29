
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def truncnorm_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays (float32)
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    a = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays with different scale and loc (float32)
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    a = np.array([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32)
    b = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32)
    loc = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    scale = np.array([[0.5, 0.5], [1.5, 1.5]], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar arrays (0D)
    x = np.array(0.5, dtype=np.float32)
    a = np.array(-3.0, dtype=np.float32)
    b = np.array(3.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 high precision arrays
    x = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    a = np.array([-1.5, -1.5, -1.5], dtype=np.float64)
    b = np.array([1.5, 1.5, 1.5], dtype=np.float64)
    loc = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    scale = np.array([1.2, 1.0, 0.8], dtype=np.float64)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x below lower bound (should result in very small value or -inf)
    x = np.array([-5.0], dtype=np.float32)
    a = np.array([-2.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x above upper bound (should result in 0.0 logcdf)
    x = np.array([5.0], dtype=np.float32)
    a = np.array([-2.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting sizes (x is 2D, others are 1D)
    x = np.random.uniform(-1, 1, size=(2, 3)).astype(np.float32)
    a = np.array([-2.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative shape bounds (a < b but both negative)
    x = np.array([-4.0, -3.5], dtype=np.float32)
    a = np.array([-5.0, -5.0], dtype=np.float32)
    b = np.array([-3.0, -3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-standard scale and loc, multi-dimensional
    x = np.arange(10).astype(np.float32).reshape(2, 5)
    a = np.full((2, 5), -3.0, dtype=np.float32)
    b = np.full((2, 5), 3.0, dtype=np.float32)
    loc = np.full((2, 5), 5.0, dtype=np.float32)
    scale = np.full((2, 5), 2.0, dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex broadcasting
    x = np.random.randn(3, 1, 4).astype(np.float32)
    a = np.array([-2.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    loc = np.random.randn(1, 2, 4).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(1, 2, 1)).astype(np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.truncnorm.logcdf"] = truncnorm_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.logcdf'.")


check_valid('jax.scipy.stats.truncnorm.logcdf', generated_inputs['jax.scipy.stats.truncnorm.logcdf'], lib="jax", suffix=0)
