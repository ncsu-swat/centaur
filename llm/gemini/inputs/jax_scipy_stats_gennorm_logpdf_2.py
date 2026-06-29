
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gennorm_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, beta=1.5
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    beta = 1.5
    list_of_inputs.append({"x": x, "beta": beta})

    # Input 2: 2D float32 array, beta=0.5
    x = np.random.randn(3, 4).astype(np.float32)
    beta = 0.5
    list_of_inputs.append({"x": x, "beta": beta})

    # Input 3: 3D float64 array, beta=2.0
    x = np.random.randn(2, 2, 2).astype(np.float64)
    beta = 2.0
    list_of_inputs.append({"x": x, "beta": beta})

    # Input 4: 0D array (scalar), beta=3.0
    x = np.array(1.5, dtype=np.float32)
    beta = 3.0
    list_of_inputs.append({"x": x, "beta": beta})

    # Input 5: 1D array with positive values, beta=0.1
    x = np.array([0.1, 0.5, 1.0, 10.0], dtype=np.float32)
    beta = 0.1
    list_of_inputs.append({"x": x, "beta": beta})

    # Input 6: 2D array of zeros, beta=1.0
    x = np.zeros((5, 5), dtype=np.float32)
    beta = 1.0
    list_of_inputs.append({"x": x, "beta": beta})

    # Input 7: 4D array of float64, beta=5.5
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 3)).astype(np.float64)
    beta = 5.5
    list_of_inputs.append({"x": x, "beta": beta})

    # Input 8: 1D array, beta=0.01 (very small positive beta)
    x = np.array([-0.5, 0.5], dtype=np.float32)
    beta = 0.01
    list_of_inputs.append({"x": x, "beta": beta})

    # Input 9: 2D array with large values, beta=10.0
    x = np.array([[-100.0, 100.0], [-50.0, 50.0]], dtype=np.float32)
    beta = 10.0
    list_of_inputs.append({"x": x, "beta": beta})

    # Input 10: 1D array with float32 values, beta=100.0 (large beta)
    x = np.random.uniform(-1.0, 1.0, size=(10,)).astype(np.float32)
    beta = 100.0
    list_of_inputs.append({"x": x, "beta": beta})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gennorm.logpdf_2"] = gennorm_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.logpdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.logpdf_2'.")


check_valid('jax.scipy.stats.gennorm.logpdf', generated_inputs['jax.scipy.stats.gennorm.logpdf_2'], lib="jax", suffix=2)
