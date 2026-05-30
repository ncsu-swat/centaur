
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tanh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 with positive, negative, and zero values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array of float32 (matrix)
    x = np.array([[-1.5, 0.5], [2.5, -0.5]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar as a 0D array (float64)
    x = np.array(0.75, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array of float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array of complex64 (complex numbers support)
    x = np.array([2.0 - 5.0j, -1.0 + 1.0j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D array of integers (int32, which will be promoted to inexact dtype)
    x = np.array([[1, -2, 3], [0, 5, -4]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Large values where tanh approaches 1 or -1
    x = np.array([-100.0, -10.0, 10.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Very small values close to 0 where tanh(x) ~ x
    x = np.array([-1e-5, 0.0, 1e-5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D array of float32
    x = np.random.uniform(-1.0, 1.0, (2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array of complex128 (double precision complex)
    x = np.array([[1.0 + 2.0j, -3.0 - 4.0j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.tanh_1"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tanh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tanh_1'.")


check_valid('jax.numpy.tanh', generated_inputs['jax.numpy.tanh_1'], lib="jax", suffix=1)
