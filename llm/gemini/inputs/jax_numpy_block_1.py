
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def block_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    # Input 2: 1D int64 array with negative values
    arr = np.array([-5, -3, 0, 2, 7], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    # Input 3: 3D float64 array
    arr = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    # Input 4: 0D scalar array
    arr = np.array(42.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    # Input 5: 4D boolean array
    arr = np.array([[[[True, False], [False, True]]]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    # Input 6: 2D int32 array
    arr = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    # Input 7: 1D float32 array
    arr = np.array([0.0, 1.5, -2.5, 3.14], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    # Input 8: 3D empty array with zero dimension
    arr = np.zeros((0, 2, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    # Input 9: 5D int32 array
    arr = np.ones((1, 2, 1, 3, 2), dtype=np.int32) * -10
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    # Input 10: 2D float64 array
    arr = np.array([[-1.5, 0.0], [2.5, -3.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arrays": arr}))

    return list_of_inputs


generated_inputs["jax.numpy.block_1"] = block_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.block_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.block_1'.")


check_valid('jax.numpy.block', generated_inputs['jax.numpy.block_1'], lib="jax", suffix=1)
