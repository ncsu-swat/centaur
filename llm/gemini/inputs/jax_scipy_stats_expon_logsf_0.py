
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expon_logsf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard values
    x = np.array(2.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 1D arrays of the same shape
    x = np.array([0.5, 1.0, 2.5, 5.0], dtype=np.float32)
    loc = np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: Float64 precision
    x = np.array([1.2, 3.4, 5.6], dtype=np.float64)
    loc = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    scale = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 2D arrays with broadcasting (x is 2D, loc is 1D, scale is scalar)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    loc = np.array([0.0, 1.0], dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Values where x < loc (testing boundary/support behavior)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Negative location parameter
    x = np.array([-1.0, 0.0, 2.0, 5.0], dtype=np.float32)
    loc = np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32)
    scale = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: 3D array inputs
    x = np.ones((2, 3, 4), dtype=np.float32) * 3.0
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Broadcasting across different dimensions (3D with (3, 1, 1), (1, 4, 1), (1, 1, 2))
    x = np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float64)
    loc = np.array([[[0.0, 0.5]]], dtype=np.float64)
    scale = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Large values for tail behavior
    x = np.array([10.0, 50.0, 100.0, 500.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Float16 inputs converted to array
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    loc = np.array([0.0, 0.5, 1.0], dtype=np.float16)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float16)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.expon.logsf"] = expon_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.logsf'.")


check_valid('jax.scipy.stats.expon.logsf', generated_inputs['jax.scipy.stats.expon.logsf'], lib="jax", suffix=0)
