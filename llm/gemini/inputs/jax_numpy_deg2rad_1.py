
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def deg2rad_inputs():
    list_of_inputs = []

    # Input 1: 1D array with common positive angles
    x = np.array([0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with negative angles
    x = np.array([-360.0, -180.0, -90.0, -45.0, -30.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array
    x = np.array(180.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of float64
    x = np.array(
        [[0.0, 90.0, 180.0], [270.0, 360.0, 720.0]], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array of integers
    x = np.array([[[0, 45], [90, 135]], [[180, 225], [270, 315]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array of random floats
    x = np.random.uniform(-360.0, 360.0, size=(2, 3, 4, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array of float16
    x = np.array([15.0, 75.0, 105.0, 165.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Array with special values (inf, -inf, nan)
    x = np.array([np.nan, np.inf, -np.inf, 0.0, 180.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array with angles larger than 360 degrees
    x = np.array([720.0, 1080.0, 1440.0, -720.0, -1080.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D integer array (int64)
    x = np.arange(12, dtype=np.int64).reshape(3, 4) * 30
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.deg2rad_1"] = deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.deg2rad_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.deg2rad_1'.")


check_valid('jax.numpy.deg2rad', generated_inputs['jax.numpy.deg2rad_1'], lib="jax", suffix=1)
