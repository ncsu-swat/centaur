
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atleast_3d_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar), float32
    arys = np.array(42.0, dtype=np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 2: 1D array, int32
    arys = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 3: 2D array, float64
    arys = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 4: 3D array, bool
    arys = np.random.choice([True, False], size=(2, 2, 2))
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 5: 4D array, uint8
    arys = np.random.randint(0, 256, size=(1, 2, 3, 4), dtype=np.uint8)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 6: 0D array with negative value, int16
    arys = np.array(-10, dtype=np.int16)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 7: 1D array with negative values, float32
    arys = np.array([-1.5, -2.5, 3.5], dtype=np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 8: 2D array with a dimension of size 1, int64
    arys = np.array([[10, 20, 30]], dtype=np.int64)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 9: 3D array, complex64
    arys = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 10: 5D array, float32
    arys = np.random.randn(2, 1, 3, 1, 2).astype(np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

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
