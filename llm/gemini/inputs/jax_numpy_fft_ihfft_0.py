
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ihfft_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, even n, default-like settings
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"a": a, "n": 4, "axis": -1, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, odd n, ortho norm
    a = np.array([-1.5, 0.0, 2.5, -3.0, 4.2], dtype=np.float64)
    input_dict = {"a": a, "n": 5, "axis": 0, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, axis 0, forward norm
    a = np.random.randn(6, 4).astype(np.float32)
    input_dict = {"a": a, "n": 6, "axis": 0, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int32 array, axis 1, n truncated
    a = np.array([[1, -2, 3, -4], [5, 6, -7, 8]], dtype=np.int32)
    input_dict = {"a": a, "n": 3, "axis": 1, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, n padded
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"a": a, "n": 8, "axis": -1, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 array, middle axis
    a = np.random.randn(4, 5, 6).astype(np.float64)
    input_dict = {"a": a, "n": 5, "axis": 1, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array with negative values
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0], dtype=np.float32)
    input_dict = {"a": a, "n": 6, "axis": 0, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, negative axis
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"a": a, "n": 4, "axis": -2, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array with zero values, odd n
    a = np.zeros((7, 9), dtype=np.float64)
    input_dict = {"a": a, "n": 7, "axis": -2, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array, n larger than array length
    a = np.array([10.0, -20.0, 30.0], dtype=np.float32)
    input_dict = {"a": a, "n": 10, "axis": -1, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D float32 array, first axis
    a = np.random.uniform(-5.0, 5.0, size=(5, 3, 2)).astype(np.float32)
    input_dict = {"a": a, "n": 6, "axis": 0, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fft.ihfft"] = ihfft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ihfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ihfft'.")


check_valid('jax.numpy.fft.ihfft', generated_inputs['jax.numpy.fft.ihfft'], lib="jax", suffix=0)
