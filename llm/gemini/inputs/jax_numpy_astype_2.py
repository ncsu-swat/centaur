
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def astype_inputs():
    list_of_inputs = []

    # Input 1: 1D float array to int32, copy=False
    x = np.array([1.5, -2.3, 0.0, 4.7], dtype=np.float32)
    input_dict = {"x": x, "dtype": "int32", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int array to float32, copy=True
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"x": x, "dtype": "float32", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D negative float array to int64, copy=False
    x = np.array([[[-1.1, -2.2], [-3.3, -4.4]]], dtype=np.float64)
    input_dict = {"x": x, "dtype": "int64", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar (0D array) to bool, copy=True
    x = np.array(0, dtype=np.int32)
    input_dict = {"x": x, "dtype": "bool", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D bool array to float64, copy=False
    x = np.array([[[[True, False]]]], dtype=np.bool_)
    input_dict = {"x": x, "dtype": "float64", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D large float array to float16, copy=True
    x = np.array([100.0, -200.0, 0.5], dtype=np.float32)
    input_dict = {"x": x, "dtype": "float16", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float array to uint32, copy=False
    x = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    input_dict = {"x": x, "dtype": "uint32", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int array to float32, copy=True
    x = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {"x": x, "dtype": "float32", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D negative int array to uint8, copy=True
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int8)
    input_dict = {"x": x, "dtype": "uint8", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float array to int32, copy=False
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict = {"x": x, "dtype": "int32", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.astype_2"] = astype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.astype_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.astype_2'.")


check_valid('jax.numpy.astype', generated_inputs['jax.numpy.astype_2'], lib="jax", suffix=2)
