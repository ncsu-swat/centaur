
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_unravel_index_inputs():
    list_of_inputs = []

    # Input 1: Scalar positive integer, 2D shape
    input_dict = {
        "indices": 5,
        "shape": (3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar negative integer, 2D shape
    input_dict = {
        "indices": -2,
        "shape": (3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D numpy array of int32, 2D shape
    input_dict = {
        "indices": np.array([1, 3, 5], dtype=np.int32),
        "shape": (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D numpy array of int64 with negative and out-of-bound elements, 2D shape
    input_dict = {
        "indices": np.array([-1, 0, 10], dtype=np.int64),
        "shape": (3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D numpy array of int32, 3D shape
    input_dict = {
        "indices": np.array([[0, 4], [8, 12]], dtype=np.int32),
        "shape": (2, 3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar integer, 1D shape
    input_dict = {
        "indices": 2,
        "shape": (5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D numpy array of int16, 4D shape
    input_dict = {
        "indices": np.array([0, 15, 30], dtype=np.int16),
        "shape": (2, 2, 2, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D numpy array of int32, 2D shape
    indices_8 = np.random.randint(-10, 20, size=(2, 2, 2), dtype=np.int32)
    input_dict = {
        "indices": indices_8,
        "shape": (4, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar numpy int64, 3D shape
    input_dict = {
        "indices": np.int64(14),
        "shape": (3, 3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D numpy array of int32, 5D shape
    input_dict = {
        "indices": np.array([100, 200, -50], dtype=np.int32),
        "shape": (2, 3, 4, 5, 6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unravel_index_3"] = jax_numpy_unravel_index_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unravel_index_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unravel_index_3'.")


check_valid('jax.numpy.unravel_index', generated_inputs['jax.numpy.unravel_index_3'], lib="jax", suffix=3)
