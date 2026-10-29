
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ifftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, odd length, axis 0
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "axes": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, even length, axis 0
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "axes": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D complex64 array, negative axis -1
    x = np.array([1.0 + 2.0j, -2.0 + 0.5j, 3.0 - 1.0j], dtype=np.complex64)
    input_dict = {"x": x, "axes": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, axis 0
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    input_dict = {"x": x, "axes": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, axis 1
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    input_dict = {"x": x, "axes": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex128 array, negative axis -2
    x = np.random.randn(4, 6).astype(np.complex128)
    input_dict = {"x": x, "axes": int(-2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int32 array, axis 2
    x = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    input_dict = {"x": x, "axes": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 array, axis -1
    x = np.random.randn(5, 5, 5).astype(np.float64)
    input_dict = {"x": x, "axes": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array, axis 3
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"x": x, "axes": int(3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D complex64 array, negative axis -4
    x = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex64)
    input_dict = {"x": x, "axes": int(-4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D int64 array, axis -1
    x = np.array([[-10, 20, -30], [40, -50, 60]], dtype=np.int64)
    input_dict = {"x": x, "axes": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.ifftshift_1"] = ifftshift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifftshift_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifftshift_1'.")


check_valid('jax.numpy.fft.ifftshift', generated_inputs['jax.numpy.fft.ifftshift_1'], lib="jax", suffix=1)
