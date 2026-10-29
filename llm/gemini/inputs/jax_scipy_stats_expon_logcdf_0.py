
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expon_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard float32
    x = np.array(2.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays, float32
    x = np.array([0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 2.0, 2.0, 0.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, float64
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    loc = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    scale = np.array([[1.5, 2.0], [0.8, 1.2]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 3D arrays, float32
    x = np.ones((2, 3, 4), dtype=np.float32) * 5.0
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 2.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - x is (4, 1), loc is (1, 3), scale is (1,)
    x = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    scale = np.array([2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: Negative loc values
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    loc = np.array([-3.0, -2.0, -1.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Values where x < loc (yielding -inf in log space)
    x = np.array([-1.0, 0.0, 0.5], dtype=np.float32)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Large scale and large x values
    x = np.array([100.0, 500.0, 1000.0], dtype=np.float64)
    loc = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    scale = np.array([200.0, 500.0, 1000.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Small positive scales and x very close to loc
    x = np.array([0.001, 0.002, 0.005], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.01, 0.02, 0.05], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 4D arrays, float32
    x = np.full((2, 2, 2, 2), 3.0, dtype=np.float32)
    loc = np.full((2, 2, 2, 2), 1.0, dtype=np.float32)
    scale = np.full((2, 2, 2, 2), 0.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.expon.logcdf"] = expon_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.logcdf'.")


check_valid('jax.scipy.stats.expon.logcdf', generated_inputs['jax.scipy.stats.expon.logcdf'], lib="jax", suffix=0)
