
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atleast_2d_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 array
    arys = np.array(3.14, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 2: 0D scalar int32 array with negative value
    arys = np.array(-42, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 3: 1D int64 array
    arys = np.array([1, 2, 3, 4], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 4: 1D empty float32 array
    arys = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 5: 1D float64 array with negative and positive values
    arys = np.array([-1.5, 0.0, 2.5, -3.75], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 6: 2D boolean array
    arys = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 7: 2D float32 array
    arys = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 8: 2D empty array
    arys = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 9: 3D uint8 array
    arys = np.zeros((2, 3, 4), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 10: 3D complex64 array
    arys = np.array([[[1.0 + 2.0j, 3.0 - 4.0j]], [[-5.0j, 6.0 + 0.0j]]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 11: 4D int16 array
    arys = np.ones((1, 2, 1, 3), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    return list_of_inputs


generated_inputs["jax.numpy.atleast_2d"] = atleast_2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_2d'.")


check_valid('jax.numpy.atleast_2d', generated_inputs['jax.numpy.atleast_2d'], lib="jax", suffix=0)
