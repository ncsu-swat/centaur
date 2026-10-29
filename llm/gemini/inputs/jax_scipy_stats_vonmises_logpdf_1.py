
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vonmises_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    x = np.array(0.5, dtype=np.float32)
    kappa = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 2: 1D arrays with matching shapes
    x = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi], dtype=np.float32)
    kappa = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 3: float64 precision
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    kappa = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 4: kappa = 0 (uniform on circle)
    x = np.linspace(-np.pi, np.pi, 7, dtype=np.float32)
    kappa = np.zeros(7, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 5: 2D arrays
    x = np.random.uniform(-np.pi, np.pi, size=(3, 4)).astype(np.float32)
    kappa = np.random.uniform(0.1, 5.0, size=(3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 6: Broadcasting (1D kappa broadcast against 2D x)
    x = np.random.uniform(-np.pi, np.pi, size=(4, 5)).astype(np.float32)
    kappa = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 7: Broadcasting (scalar-like 1D kappa broadcast against 3D x)
    x = np.random.uniform(-np.pi, np.pi, size=(2, 3, 4)).astype(np.float64)
    kappa = np.array([2.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 8: High concentration kappa values
    x = np.array([0.0, 0.01, -0.01], dtype=np.float32)
    kappa = np.array([50.0, 100.0, 200.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 9: 3D arrays with matching shapes
    x = np.random.uniform(-np.pi, np.pi, size=(2, 2, 2)).astype(np.float32)
    kappa = np.random.uniform(0.5, 10.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 10: Negative and positive angles beyond [-pi, pi]
    x = np.array([-3 * np.pi, -2 * np.pi, 2 * np.pi, 3 * np.pi], dtype=np.float32)
    kappa = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    # Input 11: Empty array handling
    x = np.array([], dtype=np.float32)
    kappa = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "kappa": kappa}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.vonmises.logpdf_1"] = vonmises_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.vonmises.logpdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.vonmises.logpdf_1'.")


check_valid('jax.scipy.stats.vonmises.logpdf', generated_inputs['jax.scipy.stats.vonmises.logpdf_1'], lib="jax", suffix=1)
