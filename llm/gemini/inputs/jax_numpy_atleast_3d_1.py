
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

    # Input 2: 0D scalar int32 with negative value
    arys = np.array(-42, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 3: 1D array float64
    arys = np.array([-1.5, 0.0, 2.5, 5.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 4: 1D empty array
    arys = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 5: 2D array int64
    arys = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 6: 2D array boolean
    arys = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 7: 3D array float32
    arys = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 8: 4D array float32
    arys = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 9: 5D array uint8
    arys = np.zeros((1, 2, 1, 3, 1), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 10: 1D complex array
    arys = np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 11: 2D single-element array
    arys = np.array([[100]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    # Input 12: 3D array with dimension size 1
    arys = np.ones((1, 5, 1), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arys": arys}))

    return list_of_inputs


generated_inputs["jax.numpy.atleast_3d_1"] = atleast_3d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_3d_1'.")


check_valid('jax.numpy.atleast_3d', generated_inputs['jax.numpy.atleast_3d_1'], lib="jax", suffix=1)
