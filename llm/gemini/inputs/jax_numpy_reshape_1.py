
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 1D, shape 6, C order
    a = np.arange(6, dtype=np.int32)
    input_dict = {"a": a, "shape": 6, "order": "C", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D to 1D, shape -1 (inferred as 6), C order
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    input_dict = {"a": a, "shape": -1, "order": "C", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D to 1D, shape 8, F order
    a = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    input_dict = {"a": a, "shape": 8, "order": "F", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D to 1D, shape -1 (inferred as 12), F order
    a = np.arange(12, dtype=np.int64)
    input_dict = {"a": a, "shape": -1, "order": "F", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D to 1D, shape 10, C order, boolean dtype
    a = np.array([True, False] * 5, dtype=np.bool_)
    input_dict = {"a": a, "shape": 10, "order": "C", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D to 1D, shape -1, F order
    a = np.ones((2, 2, 2, 2), dtype=np.float32)
    input_dict = {"a": a, "shape": -1, "order": "F", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D to 1D, shape 1, C order
    a = np.array([42.0], dtype=np.float32)
    input_dict = {"a": a, "shape": 1, "order": "C", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D to 1D, shape 24, F order
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"a": a, "shape": 24, "order": "F", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D to 1D, shape -1, C order
    a = np.random.randint(0, 10, size=(5, 5)).astype(np.int32)
    input_dict = {"a": a, "shape": -1, "order": "C", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D to 1D, shape 100, F order
    a = np.arange(100, dtype=np.float32)
    input_dict = {"a": a, "shape": 100, "order": "F", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.reshape_1"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reshape_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reshape_1'.")


check_valid('jax.numpy.reshape', generated_inputs['jax.numpy.reshape_1'], lib="jax", suffix=1)
