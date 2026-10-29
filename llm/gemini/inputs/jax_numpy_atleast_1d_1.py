
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    arys = np.array(3.14, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 2: 0D scalar int32 (negative)
    arys = np.array(-42, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 3: 0D scalar bool
    arys = np.array(True, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 4: 0D scalar complex64
    arys = np.array(1.0 - 2.0j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 5: 1D array float32
    arys = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 6: 1D empty array
    arys = np.empty((0,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 7: 1D array int64
    arys = np.arange(-5, 5, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 8: 2D array float64
    arys = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 9: 2D array uint8
    arys = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 10: 3D array int16
    arys = np.arange(8, dtype=np.int16).reshape((2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 11: 4D array float32 with singleton dimensions
    arys = np.zeros((1, 3, 1, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 12: 5D array complex128
    arys = np.ones((2, 1, 2, 1, 2), dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    return list_of_inputs


generated_inputs["jax.numpy.atleast_1d_1"] = atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_1d_1'.")


check_valid('jax.numpy.atleast_1d', generated_inputs['jax.numpy.atleast_1d_1'], lib="jax", suffix=1)
