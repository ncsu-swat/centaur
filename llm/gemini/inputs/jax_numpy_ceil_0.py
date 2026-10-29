
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ceil_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative numbers
    x = np.array([-1.7, -1.5, -0.2, 0.0, 0.2, 1.5, 1.7, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array
    x = np.array([[-3.4, 2.1, 0.0], [5.9, -0.01, -10.8]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array (tensor)
    x = np.array(4.25, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D float16 array
    x = np.array([-2.5, -0.5, 0.5, 2.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D integer array (int32)
    x = np.array([[-5, 0], [3, 10]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D unsigned integer array (uint8)
    x = np.array([0, 1, 10, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Array with inf and nan values
    x = np.array([-np.inf, -1.5, 0.0, np.nan, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D boolean array
    x = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: 1D int64 array with extreme values
    x = np.array([-9223372036854775807, -1, 0, 1, 9223372036854775806], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.ceil"] = ceil_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ceil'.")


check_valid('jax.numpy.ceil', generated_inputs['jax.numpy.ceil'], lib="jax", suffix=0)
