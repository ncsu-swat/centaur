
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nextafter_inputs():
    list_of_inputs = []

    # Input 1: Positive python integers
    input_dict = {"x": 1, "y": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative python integers
    input_dict = {"x": -5, "y": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero and negative python integers
    input_dict = {"x": 0, "y": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numpy int32 scalars
    input_dict = {"x": np.int32(10), "y": np.int32(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Numpy int64 scalars
    input_dict = {"x": np.int64(-100), "y": np.int64(100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D numpy integer arrays (int32)
    input_dict = {
        "x": np.array([1, -2, 3], dtype=np.int32),
        "y": np.array([2, -1, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D numpy integer arrays (int64)
    input_dict = {
        "x": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "y": np.array([[2, 1], [4, 3]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar and 1D array broadcasting
    input_dict = {
        "x": np.array([10, 20, 30], dtype=np.int32),
        "y": np.int32(15)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D numpy integer arrays (int32)
    input_dict = {
        "x": np.ones((2, 2, 2), dtype=np.int32),
        "y": np.zeros((2, 2, 2), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D numpy integer arrays with broadcasting
    input_dict = {
        "x": np.random.randint(-10, 10, size=(1, 3, 1, 2), dtype=np.int32),
        "y": np.random.randint(-10, 10, size=(2, 3, 4, 2), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nextafter_3"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nextafter_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nextafter_3'.")


check_valid('jax.numpy.nextafter', generated_inputs['jax.numpy.nextafter_3'], lib="jax", suffix=3)
