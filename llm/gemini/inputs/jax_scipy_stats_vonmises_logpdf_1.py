
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vonmises_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays (float32)
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    kappa = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 2: 1D arrays with float64 and different kappa values
    x = np.array([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi], dtype=np.float64)
    kappa = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 3: 2D arrays (3x3)
    x = np.random.uniform(-np.pi, np.pi, size=(3, 3)).astype(np.float32)
    kappa = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 4: Scalar arrays (0D)
    x = np.array(0.5, dtype=np.float32)
    kappa = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 5: Broad-casting case, 2D 'x' and 1D 'kappa'
    x = np.random.uniform(-np.pi, np.pi, size=(2, 4)).astype(np.float32)
    kappa = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 6: Broad-casting case, 1D 'x' and 2D 'kappa'
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    kappa = np.random.uniform(1.0, 5.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 7: High dimensions (3D arrays)
    x = np.random.uniform(-np.pi, np.pi, size=(2, 2, 2)).astype(np.float32)
    kappa = np.random.uniform(0.1, 3.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 8: kappa = 0 (uniform distribution)
    x = np.linspace(-np.pi, np.pi, 5, dtype=np.float32)
    kappa = np.zeros(5, dtype=np.float32)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 9: Large kappa values (high concentration)
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    kappa = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 10: x values outside the standard [-pi, pi] interval (periodicity test)
    x = np.array([-3*np.pi, -2*np.pi, 2*np.pi, 3*np.pi], dtype=np.float32)
    kappa = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "kappa": kappa})

    # Input 11: Single element array with float64
    x = np.array([1.23], dtype=np.float64)
    kappa = np.array([0.88], dtype=np.float64)
    list_of_inputs.append({"x": x, "kappa": kappa})

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
