
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis=0, keepdims=False
    a = np.array([1.0, 3.0, 5.0, 4.0, 2.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32, axis=0, keepdims=True
    a = np.array([[1.0, 3.0, 2.0], [5.0, 4.0, 1.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, int32, axis=1, keepdims=False
    a = np.array([[10, 20, 30], [60, 50, 40]], dtype=np.int32)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, float64, axis=2, keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"a": a, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, int64, axis=-1, keepdims=False
    a = np.random.randint(-100, 100, size=(2, 2, 3, 3)).astype(np.int64)
    input_dict = {"a": a, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with duplicate max values, float32, axis=1, keepdims=True
    a = np.array([[1.5, 3.5, 3.5], [2.0, 2.0, 1.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with negative values, int32, axis=0, keepdims=False
    a = np.array([-10, -5, -2, -50, -1], dtype=np.int32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, float32, axis=0, keepdims=False
    a = np.random.uniform(-1.0, 1.0, (3, 3, 3)).astype(np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, boolean, axis=0, keepdims=True
    a = np.array([[True, False, True], [False, False, True]], dtype=bool)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, float32, axis=3, keepdims=False
    a = np.random.randn(2, 2, 2, 4, 2).astype(np.float32)
    input_dict = {"a": a, "axis": 3, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.argmax"] = argmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argmax'.")


check_valid('jax.numpy.argmax', generated_inputs['jax.numpy.argmax'], lib="jax", suffix=0)
