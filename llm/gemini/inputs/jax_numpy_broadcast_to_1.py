
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_to_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array to 1D shape
    array = np.array(5, dtype=np.int32)
    shape = (4,)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 2: 1D array to 2D shape
    array = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    shape = (4, 3)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 3: 2D column vector to 2D matrix
    array = np.array([[-10], [20], [-30]], dtype=np.int64)
    shape = (3, 5)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 4: 2D row vector to 3D tensor
    array = np.array([[1.5, -2.5, 0.0, 4.2]], dtype=np.float64)
    shape = (2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 5: Boolean array broadcasting
    array = np.array([[True], [False]], dtype=np.bool_)
    shape = (2, 4)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 6: Higher-dimensional (3D to 4D) broadcasting
    array = np.ones((1, 4, 1), dtype=np.int16)
    shape = (2, 3, 4, 5)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 7: Complex numbers broadcasting
    array = np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)
    shape = (3, 2)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 8: 0D scalar to 3D tensor
    array = np.array(-1.23, dtype=np.float32)
    shape = (2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 9: Unchanged shape (identity broadcast)
    array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    shape = (2, 2)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 10: Array with size 0 dimension broadcasted
    array = np.empty((0, 1), dtype=np.float32)
    shape = (0, 5)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

    # Input 11: 4D to 5D broadcasting with multiple singleton dimensions
    array = np.random.randn(1, 3, 1, 4).astype(np.float32)
    shape = (2, 5, 3, 7, 4)
    list_of_inputs.append(copy.deepcopy({"array": array, "shape": shape}))

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
