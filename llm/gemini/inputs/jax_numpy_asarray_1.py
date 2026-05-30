
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def asarray_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, copy=True, order='K'
    a = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    input_dict = {"a": a, "dtype": np.float32, "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, copy=False, order='K'
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"a": a, "dtype": np.int32, "order": "K", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, copy=True, order='K'
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"a": a, "dtype": np.float32, "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D bool array, copy=False, order='K'
    a = np.array([True, False, True], dtype=bool)
    input_dict = {"a": a, "dtype": np.bool_, "order": "K", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D int64 array, copy=True, order='K'
    a = np.arange(16).reshape(2, 2, 2, 2).astype(np.int64)
    input_dict = {"a": a, "dtype": np.int64, "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex64 array, copy=False, order='K'
    a = np.array([[1 + 2j, 3 - 4j]], dtype=np.complex64)
    input_dict = {"a": a, "dtype": np.complex64, "order": "K", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D uint8 array, copy=True, order='K'
    a = np.array([0, 128, 255], dtype=np.uint8)
    input_dict = {"a": a, "dtype": np.float64, "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array with negative values, copy=False, order='K'
    a = np.random.uniform(-10.0, 10.0, size=(3, 3, 3)).astype(np.float32)
    input_dict = {"a": a, "dtype": np.float32, "order": "K", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int16 array, copy=True, order='K'
    a = np.array([-32768, 0, 32767], dtype=np.int16)
    input_dict = {"a": a, "dtype": np.int32, "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, copy=False, order='K'
    a = np.ones((1, 2, 1, 3, 1), dtype=np.float32)
    input_dict = {"a": a, "dtype": np.float32, "order": "K", "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.asarray_1"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asarray_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asarray_1'.")


check_valid('jax.numpy.asarray', generated_inputs['jax.numpy.asarray_1'], lib="jax", suffix=1)
