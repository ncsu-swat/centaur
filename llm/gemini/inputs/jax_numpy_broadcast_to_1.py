
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_to_inputs():
    list_of_inputs = []

    # Input 1: Scalar to 1D
    array = np.array(4.2, dtype=np.float32)
    shape = (5,)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 2: 1D to 2D
    array = np.array([1, 2, 3], dtype=np.int32)
    shape = (2, 3)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 3: Column vector (2D) to 2D Matrix
    array = np.arange(4, dtype=np.int64).reshape(4, 1)
    shape = (4, 5)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 4: Row vector (2D) to 2D Matrix
    array = np.arange(5, dtype=np.float64).reshape(1, 5)
    shape = (3, 5)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 5: 1D with negative values to 3D
    array = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    shape = (4, 2, 3)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 6: 2D to 3D
    array = np.random.randn(3, 1).astype(np.float32)
    shape = (2, 3, 5)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 7: 3D to 3D
    array = np.random.randn(1, 4, 1).astype(np.float32)
    shape = (2, 4, 3)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 8: Boolean array
    array = np.array([[True], [False]], dtype=np.bool_)
    shape = (2, 3)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 9: Complex array
    array = np.array([1+2j, 3-4j], dtype=np.complex64)
    shape = (2, 2)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 10: High dimensional broadcasting
    array = np.ones((1, 1, 1, 5), dtype=np.int16)
    shape = (2, 3, 4, 5)
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    return list_of_inputs

generated_inputs["jax.numpy.broadcast_to_1"] = broadcast_to_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.broadcast_to_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.broadcast_to_1'.")


check_valid('jax.numpy.broadcast_to', generated_inputs['jax.numpy.broadcast_to_1'], lib="jax", suffix=1)
