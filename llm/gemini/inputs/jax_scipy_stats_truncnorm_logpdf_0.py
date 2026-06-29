
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_truncnorm_logpdf_inputs():
    list_of_inputs = []

    # 1. Standard 1D arrays, float32, identical shapes
    x = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    a = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. 0D scalar arrays
    x = np.array(0.0, dtype=np.float32)
    a = np.array(-2.0, dtype=np.float32)
    b = np.array(2.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. float64, 2D arrays
    x = np.array([[0.1, 0.2], [-0.1, -0.2]], dtype=np.float64)
    a = np.array([[-3.0, -3.0], [-3.0, -3.0]], dtype=np.float64)
    b = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float64)
    loc = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float64)
    scale = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. Broadcasting, mix of shapes
    x = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
    a = np.array([[-2.0, -1.0, -0.5]], dtype=np.float32)
    b = np.array([[2.0, 1.0, 0.5]], dtype=np.float32)
    loc = np.array([[0.0]], dtype=np.float32)
    scale = np.array([[1.0]], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. Outside bounds (value of x will fall outside standard range)
    x = np.array([5.0], dtype=np.float32)
    a = np.array([-2.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. High scale and small scale values
    x = np.array([10.0, 0.01], dtype=np.float32)
    a = np.array([-2.0, -2.0], dtype=np.float32)
    b = np.array([2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([10.0, 0.01], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. 3D arrays
    x = np.zeros((2, 2, 2), dtype=np.float32)
    a = -np.ones((2, 2, 2), dtype=np.float32)
    b = np.ones((2, 2, 2), dtype=np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. Unsymmetric bounds
    x = np.array([0.5, -0.1], dtype=np.float32)
    a = np.array([-0.5, -1.5], dtype=np.float32)
    b = np.array([2.5, 0.5], dtype=np.float32)
    loc = np.array([0.1, -0.2], dtype=np.float32)
    scale = np.array([0.8, 1.2], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. Semi-infinite truncation representation (b is large)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. High-dimensional 4D arrays, float64
    x = np.random.uniform(-1, 1, size=(2, 1, 3, 2)).astype(np.float64)
    a = np.full((2, 1, 3, 2), -2.0, dtype=np.float64)
    b = np.full((2, 1, 3, 2), 2.0, dtype=np.float64)
    loc = np.full((2, 1, 3, 2), 0.0, dtype=np.float64)
    scale = np.full((2, 1, 3, 2), 1.0, dtype=np.float64)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.truncnorm.logpdf"] = jax_scipy_stats_truncnorm_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.logpdf'.")


check_valid('jax.scipy.stats.truncnorm.logpdf', generated_inputs['jax.scipy.stats.truncnorm.logpdf'], lib="jax", suffix=0)
