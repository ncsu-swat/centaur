
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logistic_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D array, standard logistic distribution
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = 0.0
    scale = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 2: 1D array with float64, shifted loc and scaled
    x = np.linspace(-5.0, 5.0, 11, dtype=np.float64)
    loc = 2.5
    scale = 0.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 3: 2D array, negative loc
    x = np.random.randn(3, 4).astype(np.float32)
    loc = -1.5
    scale = 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 4: 0D scalar array
    x = np.array(0.0, dtype=np.float32)
    loc = 0.0
    scale = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 5: 3D array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    loc = 1.0
    scale = 3.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 6: Large positive values in x
    x = np.array([10.0, 50.0, 100.0, 500.0], dtype=np.float32)
    loc = 0.0
    scale = 10.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 7: Large negative values in x
    x = np.array([-500.0, -100.0, -50.0, -10.0], dtype=np.float64)
    loc = -5.0
    scale = 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 8: 4D array
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    loc = 0.5
    scale = 1.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 9: Small positive scale
    x = np.array([-0.01, 0.0, 0.01], dtype=np.float64)
    loc = 0.0
    scale = 0.01
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    loc = 0.0
    scale = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 11: 2D array with large scale
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    loc = 10.0
    scale = 100.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.logistic.logpdf_2"] = logistic_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.logpdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.logpdf_2'.")


check_valid('jax.scipy.stats.logistic.logpdf', generated_inputs['jax.scipy.stats.logistic.logpdf_2'], lib="jax", suffix=2)
