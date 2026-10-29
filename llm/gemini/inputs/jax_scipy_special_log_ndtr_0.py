
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log_ndtr_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, series_order=3 (default value)
    x = np.array([-15.0, -5.0, 0.0, 3.0, 10.0], dtype=np.float32)
    series_order = 3
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 2: 1D array, float64, series_order=5
    x = np.array([-25.0, -10.0, 0.0, 5.0, 12.0], dtype=np.float64)
    series_order = 5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 3: Scalar (0D tensor), float32, series_order=0
    x = np.array(-2.5, dtype=np.float32)
    series_order = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 4: 2D array, float32, series_order=10
    x = np.random.randn(3, 4).astype(np.float32)
    series_order = 10
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 5: 2D array, float64, series_order=30 (maximum allowed order)
    x = np.linspace(-30.0, 10.0, 12, dtype=np.float64).reshape(4, 3)
    series_order = 30
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 6: 3D array, float32, series_order=1
    x = np.random.uniform(-15.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    series_order = 1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 7: 4D array, float64, series_order=7
    x = np.random.randn(2, 2, 3, 2).astype(np.float64)
    series_order = 7
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 8: All positive values, float32, series_order=4
    x = np.array([1.0, 2.0, 5.0, 10.0, 20.0], dtype=np.float32)
    series_order = 4
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 9: Strongly negative values (tests asymptotic branch), float64, series_order=15
    x = np.array([-100.0, -50.0, -35.0, -22.0], dtype=np.float64)
    series_order = 15
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 10: Empty 1D array, float32, series_order=3
    x = np.array([], dtype=np.float32)
    series_order = 3
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    # Input 11: Zeros array, float64, series_order=2
    x = np.zeros((5, 5), dtype=np.float64)
    series_order = 2
    list_of_inputs.append(
        copy.deepcopy({"x": x, "series_order": series_order})
    )

    return list_of_inputs


generated_inputs["jax.scipy.special.log_ndtr"] = log_ndtr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.log_ndtr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.log_ndtr'.")


check_valid('jax.scipy.special.log_ndtr', generated_inputs['jax.scipy.special.log_ndtr'], lib="jax", suffix=0)
