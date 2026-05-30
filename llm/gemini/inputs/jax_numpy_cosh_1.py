
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cosh_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar) float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array float32 with positive, negative, and zero values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array float64
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array float32
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array complex64
    x = np.array([1.0 + 1.0j, -2.0 - 3.0j, 0.5j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D array complex128
    real = np.random.randn(2, 2)
    imag = np.random.randn(2, 2)
    x = (real + 1j * imag).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array with very small values
    x = np.array([1e-15, -1e-15, 0.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D array float32
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array containing Inf and NaN
    x = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array float16
    x = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.cosh_1"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cosh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cosh_1'.")


check_valid('jax.numpy.cosh', generated_inputs['jax.numpy.cosh_1'], lib="jax", suffix=1)
