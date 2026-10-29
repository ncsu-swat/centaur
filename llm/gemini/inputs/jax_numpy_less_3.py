
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def less_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array and zero
    x = np.array([-5, 0, 5, 10], dtype=np.int32)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float32 array and positive integer
    x = np.random.randn(3, 4).astype(np.float32)
    y = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D float64 array and negative integer
    x = np.array([-10.5, -3.2, 0.0, 4.7], dtype=np.float64)
    y = -3
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int64 array and positive integer
    x = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int64)
    y = 50
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar array and integer
    x = np.array(7, dtype=np.int32)
    y = 10
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D float32 array and negative integer
    x = np.random.uniform(-20.0, 20.0, size=(2, 2, 3, 3)).astype(np.float32)
    y = -10
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D uint8 array and integer
    x = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    y = 128
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D float32 array with inf and nan values
    x = np.array([-np.inf, np.nan, 0.0, np.inf], dtype=np.float32)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D boolean array and integer
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 3D int16 array and large integer
    x = np.random.randint(-30000, 30000, size=(1, 5, 2), dtype=np.int16)
    y = 1000
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 1D int8 array and negative integer
    x = np.array([-128, -64, 0, 64, 127], dtype=np.int8)
    y = -50
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.less_3"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_3'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_3'], lib="jax", suffix=3)
