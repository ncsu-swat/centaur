
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, expand axis 0
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array, expand axis 1
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32 array, expand axis -1
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = -1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int32 array, expand axis 0
    a = np.random.randint(0, 10, size=(3, 4)).astype(np.int32)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int32 array, expand axis 1
    a = np.random.randint(0, 10, size=(3, 4)).astype(np.int32)
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 array, expand axis 2
    a = np.random.randint(0, 10, size=(3, 4)).astype(np.int32)
    axis = 2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int32 array, expand axis -2
    a = np.random.randint(0, 10, size=(3, 4)).astype(np.int32)
    axis = -2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 array, expand axis 3
    a = np.random.randn(2, 2, 2).astype(np.float64)
    axis = 3
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 array, expand axis -4
    a = np.random.randn(2, 2, 2).astype(np.float64)
    axis = -4
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 0D float32 array, expand axis 0
    a = np.array(42.0, dtype=np.float32)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.expand_dims_1"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expand_dims_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expand_dims_1'.")


check_valid('jax.numpy.expand_dims', generated_inputs['jax.numpy.expand_dims_1'], lib="jax", suffix=1)
