
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atleast_3d_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    arys = np.array(3.14, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 2: 1D array with negative and positive int32 values
    arys = np.array([-5, -1, 0, 2, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 3: 2D float64 array
    arys = np.array([[-1.5, 2.3, 0.0], [4.1, -3.2, 5.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 4: 3D int16 array
    arys = np.ones((2, 3, 4), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 5: 4D float32 array
    arys = np.zeros((1, 2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 6: 1D boolean array
    arys = np.array([True, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 7: Empty 1D float64 array
    arys = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 8: Empty 2D int32 array
    arys = np.empty((0, 5), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 9: 1D complex128 array with negative components
    arys = np.array([1 + 2j, -3 + 4j, -0.5 - 1.5j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 10: 2D uint8 array
    arys = np.array([[255, 0], [128, 64]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 11: 5D int64 array
    arys = np.arange(24, dtype=np.int64).reshape((1, 2, 3, 2, 2))
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    return list_of_inputs


generated_inputs["jax.numpy.atleast_3d_2"] = atleast_3d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_3d_2'.")


check_valid('jax.numpy.atleast_3d', generated_inputs['jax.numpy.atleast_3d_2'], lib="jax", suffix=2)
