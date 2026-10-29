
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def concatenate_inputs():
    list_of_inputs = []

    # Input 1: 2D array unstacked along axis 0, concatenated along axis 0, float32
    arrays = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append({
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype(np.float32)
    })

    # Input 2: 2D array unstacked along axis 0, concatenated along axis -1, int32
    arrays = np.random.randint(-10, 10, size=(3, 4), dtype=np.int32)
    list_of_inputs.append({
        "arrays": arrays,
        "axis": -1,
        "dtype": np.dtype(np.int32)
    })

    # Input 3: 2D array unstacked along axis 0, concatenated along axis 0, bool
    arrays = np.random.choice([True, False], size=(4, 2))
    list_of_inputs.append({
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype(np.bool_)
    })

    # Input 4: 3D array unstacked along axis 0, concatenated along axis 0, float32
    arrays = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype(np.float32)
    })

    # Input 5: 3D array unstacked along axis 0, concatenated along axis 1, float32
    arrays = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({
        "arrays": arrays,
        "axis": 1,
        "dtype": np.dtype(np.float32)
    })

    # Input 6: 3D array unstacked along axis 0, concatenated along axis -1, int32
    arrays = np.random.randint(0, 100, size=(4, 2, 5), dtype=np.int32)
    list_of_inputs.append({
        "arrays": arrays,
        "axis": -1,
        "dtype": np.dtype(np.int32)
    })

    # Input 7: 3D array unstacked along axis 0, concatenated along axis -2, bool
    arrays = np.random.choice([True, False], size=(3, 2, 4))
    list_of_inputs.append({
        "arrays": arrays,
        "axis": -2,
        "dtype": np.dtype(np.bool_)
    })

    # Input 8: 4D array unstacked along axis 0, concatenated along axis 0, float32
    arrays = np.random.randn(2, 3, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype(np.float32)
    })

    # Input 9: 4D array unstacked along axis 0, concatenated along axis 2, int32
    arrays = np.random.randint(-50, 50, size=(2, 3, 2, 2), dtype=np.int32)
    list_of_inputs.append({
        "arrays": arrays,
        "axis": 2,
        "dtype": np.dtype(np.int32)
    })

    # Input 10: 4D array unstacked along axis 0, concatenated along axis 1, bool
    arrays = np.random.choice([True, False], size=(3, 2, 4, 3))
    list_of_inputs.append({
        "arrays": arrays,
        "axis": 1,
        "dtype": np.dtype(np.bool_)
    })

    # Input 11: 5D array unstacked along axis 0, concatenated along axis 3, float32
    arrays = np.random.randn(2, 2, 3, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "arrays": arrays,
        "axis": 3,
        "dtype": np.dtype(np.float32)
    })

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.concatenate_1"] = concatenate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.concatenate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.concatenate_1'.")


check_valid('jax.numpy.concatenate', generated_inputs['jax.numpy.concatenate_1'], lib="jax", suffix=1)
