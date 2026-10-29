
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unravel_index_inputs():
    list_of_inputs = []

    # Input 1: Basic zero index and unit shape (Python int)
    input_dict = {"indices": int(0), "shape": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Within-bounds positive index (Python int)
    input_dict = {"indices": int(3), "shape": int(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: np.int32 types within bounds
    input_dict = {"indices": np.int32(2), "shape": np.int32(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: np.int64 types within bounds
    input_dict = {"indices": np.int64(7), "shape": np.int64(12)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative index supported in JAX (Python int)
    input_dict = {"indices": int(-1), "shape": int(6)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative index with np.int32
    input_dict = {"indices": np.int32(-3), "shape": np.int32(8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Out-of-bounds positive index (Python int)
    input_dict = {"indices": int(15), "shape": int(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Out-of-bounds index with np.int64
    input_dict = {"indices": np.int64(100), "shape": np.int64(50)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Out-of-bounds negative index with np.int64
    input_dict = {"indices": np.int64(-50), "shape": np.int64(25)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.int16 types with large shape
    input_dict = {"indices": np.int16(0), "shape": np.int16(100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Larger shape with intermediate index (Python int)
    input_dict = {"indices": int(42), "shape": int(1000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Upper bound boundary index (indices == shape - 1)
    input_dict = {"indices": np.int32(19), "shape": np.int32(20)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.unravel_index_4"] = unravel_index_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unravel_index_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unravel_index_4'.")


check_valid('jax.numpy.unravel_index', generated_inputs['jax.numpy.unravel_index_4'], lib="jax", suffix=4)
