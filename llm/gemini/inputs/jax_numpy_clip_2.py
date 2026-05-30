
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def clip_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, clipping range [2.0, 5.0]
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"arr": arr, "min": 2.0, "max": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, negative values, clipping range [-1.5, 1.5]
    arr = np.random.uniform(-3.0, 3.0, size=(3, 4)).astype(np.float32)
    input_dict = {"arr": arr, "min": -1.5, "max": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, clipping range [0.1, 0.9]
    arr = np.random.rand(2, 3, 3).astype(np.float64)
    input_dict = {"arr": arr, "min": 0.1, "max": 0.9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int32 array (min/max are floats), clipping range [2.5, 7.5]
    arr = np.array([-1, 0, 5, 10, 12], dtype=np.int32)
    input_dict = {"arr": arr, "min": 2.5, "max": 7.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, clipping range [-10.0, 10.0]
    arr = (np.random.randn(2, 2, 2, 2) * 20.0).astype(np.float32)
    input_dict = {"arr": arr, "min": -10.0, "max": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float16 array, clipping range [0.0, 1.0]
    arr = np.array([-0.5, 0.5, 1.5], dtype=np.float16)
    input_dict = {"arr": arr, "min": 0.0, "max": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: min is larger than max (special case, max value is returned)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"arr": arr, "min": 3.0, "max": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar-like 1D array, clipping range [0.0, 10.0]
    arr = np.array([5.5], dtype=np.float32)
    input_dict = {"arr": arr, "min": 0.0, "max": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values, 2D float64 array, clipping range [1000.0, 5000.0]
    arr = np.array([[500.0, 1500.0], [6000.0, 2500.0]], dtype=np.float64)
    input_dict = {"arr": arr, "min": 1000.0, "max": 5000.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, clipping range [-0.5, 0.5]
    arr = np.random.uniform(-1.0, 1.0, size=(1, 2, 2, 2, 1)).astype(np.float32)
    input_dict = {"arr": arr, "min": -0.5, "max": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.clip_2"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.clip_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.clip_2'.")


check_valid('jax.numpy.clip', generated_inputs['jax.numpy.clip_2'], lib="jax", suffix=2)
