
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rad2deg_inputs():
    list_of_inputs = []

    # Input 1: Scalar python int
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar python negative int
    input_dict = {"x": -180}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Numpy scalar int32
    input_dict = {"x": np.int32(45)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numpy scalar int64 negative
    input_dict = {"x": np.int64(-90)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D numpy array of int32
    input_dict = {"x": np.array([0, 1, 2, -1, -2], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D numpy array of int64
    input_dict = {"x": np.array([[1, -2], [3, 4]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D numpy array of int16
    input_dict = {"x": np.arange(-12, 12, dtype=np.int16).reshape(2, 3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar python large int
    input_dict = {"x": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D numpy array of int8
    input_dict = {"x": np.array([-128, 0, 127], dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D numpy array of int32
    input_dict = {"x": np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D numpy array of int64
    input_dict = {"x": np.random.randint(-10, 10, size=(1, 2, 1, 2, 1), dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.rad2deg_3"] = rad2deg_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.rad2deg_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.rad2deg_3'.")


check_valid('jax.numpy.rad2deg', generated_inputs['jax.numpy.rad2deg_3'], lib="jax", suffix=3)
