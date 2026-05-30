
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_to_inputs():
    list_of_inputs = []

    # Input 1: Scalar to 2D
    array = np.array(5.0, dtype=np.float32)
    shape = [2, 3]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 2: 1D to 2D
    array = np.array([1, 2, 3], dtype=np.int32)
    shape = [2, 3]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 3: 2D to 3D with float32
    array = np.random.randn(2, 3).astype(np.float32)
    shape = [4, 2, 3]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 4: 1D to 1D (same shape)
    array = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    shape = [3]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 5: 2D with singleton dimension to 2D with float64
    array = np.random.randn(5, 1).astype(np.float64)
    shape = [5, 4]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 6: 3D with singleton dimensions to 3D with int32
    array = np.random.randint(-10, 10, size=(1, 3, 1)).astype(np.int32)
    shape = [2, 3, 5]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 7: Boolean array broadcasting
    array = np.array([[True], [False]], dtype=np.bool_)
    shape = [2, 3]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 8: Complex64 broadcasting
    array = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    shape = [2, 3]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 9: High-dimensional broadcasting
    array = np.random.randn(1, 1, 4).astype(np.float32)
    shape = [2, 3, 2, 4]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    # Input 10: Int64 broadcasting
    array = np.array([1000, 2000], dtype=np.int64)
    shape = [5, 2]
    list_of_inputs.append({"array": copy.deepcopy(array), "shape": shape})

    return list_of_inputs

generated_inputs["jax.numpy.broadcast_to_2"] = broadcast_to_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.broadcast_to_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.broadcast_to_2'.")


check_valid('jax.numpy.broadcast_to', generated_inputs['jax.numpy.broadcast_to_2'], lib="jax", suffix=2)
