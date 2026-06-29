
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def chi2_logsf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, standard df and scale
    x = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    df = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float32
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    df = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 1D arrays, non-zero location and scale
    x = np.array([1.5, 2.5], dtype=np.float64)
    df = np.array([1.0, 5.0], dtype=np.float64)
    loc = np.array([-1.0, 0.5], dtype=np.float64)
    scale = np.array([2.0, 0.5], dtype=np.float64)
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar-like 0D arrays
    x = np.array(3.0, dtype=np.float32)
    df = np.array(5.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large 3D arrays
    x = np.ones((2, 3, 4), dtype=np.float32) * 4.5
    df = np.ones((2, 3, 4), dtype=np.float32) * 3.0
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting case - 2D x, 1D others
    x = np.random.uniform(5.0, 10.0, size=(3, 3)).astype(np.float32)
    df = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High degrees of freedom
    x = np.array([100.0, 150.0], dtype=np.float32)
    df = np.array([100.0, 100.0], dtype=np.float32)
    loc = np.array([10.0, 10.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative location parameters
    x = np.array([0.0, 1.0], dtype=np.float32)
    df = np.array([2.0, 2.0], dtype=np.float32)
    loc = np.array([-2.0, -2.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small fractional df and scale
    x = np.array([0.1, 0.2], dtype=np.float32)
    df = np.array([0.5, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([0.1, 0.2], dtype=np.float32)
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting with 1-element arrays
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    df = np.array([5.0], dtype=np.float32)
    loc = np.array([2.0], dtype=np.float32)
    scale = np.array([3.0], dtype=np.float32)
    input_dict = {"x": x, "df": df, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.chi2.logsf"] = chi2_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.chi2.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.chi2.logsf'.")


check_valid('jax.scipy.stats.chi2.logsf', generated_inputs['jax.scipy.stats.chi2.logsf'], lib="jax", suffix=0)
