
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fabs_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative and positive values
    x = np.array([-1.342, 5.649, -3.927, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D integer array
    x = np.array([-5, -9, 1, 10, 15, 0], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float64 array
    x = np.array([[-2.5, 3.1], [-0.0, -100.4]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D scalar array (float32)
    x = np.array(-42.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D float16 array
    x = np.random.uniform(-10, 10, size=(2, 3, 4)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D boolean array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D int64 array
    x = np.random.randint(-100, 100, size=(2, 2, 3, 2), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Array with special float values (inf, -inf, nan)
    x = np.array([-np.inf, np.inf, np.nan, -0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D unsigned integer array
    x = np.array([[0, 10, 255], [128, 5, 20]], dtype=np.uint8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D int16 array
    x = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.numpy.fabs_1"] = fabs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fabs_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fabs_1'.")


check_valid('jax.numpy.fabs', generated_inputs['jax.numpy.fabs_1'], lib="jax", suffix=1)
