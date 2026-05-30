
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_concat_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, axis=0
    arrays = np.random.randn(2, 5).astype(np.float32)
    input_dict = {"arrays": arrays, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float32 array, axis=0
    arrays = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {"arrays": arrays, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array, axis=1
    arrays = np.random.randint(0, 10, size=(3, 4, 5)).astype(np.int32)
    input_dict = {"arrays": arrays, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, axis=-1
    arrays = np.random.randn(3, 4, 5).astype(np.float64)
    input_dict = {"arrays": arrays, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D boolean array, axis=-1
    arrays = np.random.choice([True, False], size=(2, 10))
    input_dict = {"arrays": arrays, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, axis=2
    arrays = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"arrays": arrays, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, axis=-2
    arrays = np.random.randn(4, 2, 3).astype(np.float32)
    input_dict = {"arrays": arrays, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D int64 array, axis=3
    arrays = np.random.randint(0, 10, size=(2, 3, 2, 3, 2)).astype(np.int64)
    input_dict = {"arrays": arrays, "axis": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array, axis=0
    arrays = np.random.randn(5, 2).astype(np.float64)
    input_dict = {"arrays": arrays, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int32 array, axis=0
    arrays = np.random.randint(-10, 10, size=(2, 6)).astype(np.int32)
    input_dict = {"arrays": arrays, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.concat_1"] = jax_numpy_concat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.concat_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.concat_1'.")


check_valid('jax.numpy.concat', generated_inputs['jax.numpy.concat_1'], lib="jax", suffix=1)
