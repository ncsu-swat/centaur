
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def less_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array and positive float
    x = np.array([-2.5, 0.0, 1.5, 3.2], dtype=np.float32)
    y = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float64 array and negative float
    x = np.array([[-10.0, -5.0], [0.0, 5.0]], dtype=np.float64)
    y = -2.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D int32 array and zero float
    x = np.array([-5, -1, 0, 1, 5], dtype=np.int32)
    y = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int64 array and float
    x = np.arange(12, dtype=np.int64).reshape((2, 2, 3)) - 6
    y = 2.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar array and float
    x = np.array(4.2, dtype=np.float32)
    y = 4.2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D float32 array and float
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    y = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Array with inf and nan values
    x = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan], dtype=np.float64)
    y = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D uint8 array and float
    x = np.array([0, 50, 100, 150, 200, 255], dtype=np.uint8)
    y = 127.5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Empty array
    x = np.empty((0, 5), dtype=np.float32)
    y = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D float16 array and large float
    x = np.array([[-100.0, 200.0], [50.0, -50.0]], dtype=np.float16)
    y = 1e4
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 5D array
    x = np.zeros((1, 2, 1, 3, 1), dtype=np.float32)
    y = -0.1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.less_2"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_2'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_2'], lib="jax", suffix=2)
