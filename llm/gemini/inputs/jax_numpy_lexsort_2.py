
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_lexsort_inputs():
    list_of_inputs = []

    keys = np.array([[2, 4, 2, 3], [1, 2, 1, 3]], dtype=np.int32)
    list_of_inputs.append({"keys": keys, "axis": -1})

    keys = np.random.randn(3, 10).astype(np.float32)
    list_of_inputs.append({"keys": keys, "axis": -1})

    keys = np.random.randint(-10, 10, size=(2, 4, 5)).astype(np.int32)
    list_of_inputs.append({"keys": keys, "axis": -1})

    keys = np.random.randint(0, 5, size=(2, 4, 5)).astype(np.int32)
    list_of_inputs.append({"keys": keys, "axis": 0})

    keys = np.random.uniform(-100.0, 100.0, size=(2, 8)).astype(np.float64)
    list_of_inputs.append({"keys": keys, "axis": -1})

    keys = np.array([[1, 1, 2, 2, 1, 1], [2, 2, 1, 1, 2, 2]], dtype=np.int64)
    list_of_inputs.append({"keys": keys, "axis": -1})

    keys = np.random.randint(0, 10, size=(2, 2, 3, 4)).astype(np.int32)
    list_of_inputs.append({"keys": keys, "axis": -2})

    keys = np.random.randint(0, 100, size=(3, 15)).astype(np.int32)
    list_of_inputs.append({"keys": keys, "axis": -1})

    keys = np.random.randn(4, 100).astype(np.float32)
    list_of_inputs.append({"keys": keys, "axis": -1})

    keys = np.random.randint(0, 10, size=(1, 20)).astype(np.int32)
    list_of_inputs.append({"keys": keys, "axis": -1})

    keys = np.random.randint(-5, 5, size=(2, 8)).astype(np.int32)
    list_of_inputs.append({"keys": keys, "axis": -1})

    return list_of_inputs

generated_inputs["jax.numpy.lexsort_2"] = jax_numpy_lexsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.lexsort_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.lexsort_2'.")


check_valid('jax.numpy.lexsort', generated_inputs['jax.numpy.lexsort_2'], lib="jax", suffix=2)
