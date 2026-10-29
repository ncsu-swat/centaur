
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def concat_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, axis 0
    arrays = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"arrays": arrays, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, negative axis -1
    arrays = np.random.randint(-10, 10, size=(4, 5), dtype=np.int32)
    input_dict = {"arrays": arrays, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, axis 0
    arrays = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"arrays": arrays, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D boolean array, axis 1
    arrays = np.random.choice([True, False], size=(3, 2, 5))
    input_dict = {"arrays": arrays, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, negative axis -2
    arrays = np.random.randint(-100, 100, size=(4, 3, 2), dtype=np.int64)
    input_dict = {"arrays": arrays, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, axis 2
    arrays = np.random.randn(2, 3, 2, 2).astype(np.float32)
    input_dict = {"arrays": arrays, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D uint8 array, negative axis -1
    arrays = np.random.randint(0, 255, size=(3, 2, 4, 3), dtype=np.uint8)
    input_dict = {"arrays": arrays, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D complex64 array, axis 0
    real = np.random.randn(2, 2, 2, 2).astype(np.float32)
    imag = np.random.randn(2, 2, 2, 2).astype(np.float32)
    arrays = real + 1j * imag
    input_dict = {"arrays": arrays, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float16 array, axis 3
    arrays = np.random.randn(2, 2, 3, 2, 4).astype(np.float16)
    input_dict = {"arrays": arrays, "axis": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int16 array, negative axis -3
    arrays = np.random.randint(-50, 50, size=(3, 1, 2, 1, 2), dtype=np.int16)
    input_dict = {"arrays": arrays, "axis": -3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array with single element leading dimension
    arrays = np.random.randn(1, 10).astype(np.float32)
    input_dict = {"arrays": arrays, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 3D float32 array, axis -1
    arrays = np.random.randn(5, 4, 3).astype(np.float32)
    input_dict = {"arrays": arrays, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.concat_2"] = concat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.concat_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.concat_2'.")


check_valid('jax.numpy.concat', generated_inputs['jax.numpy.concat_2'], lib="jax", suffix=2)
