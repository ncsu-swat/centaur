
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
from jax.sharding import Mesh, PartitionSpec

# Initialize and enter a global mesh context for the current thread
# using the non-deprecated jax.set_mesh context manager.
devices = jax.devices()
mesh = Mesh(np.array(devices), ('x',))
ctx = jax.set_mesh(mesh)
ctx.__enter__()

def zeros_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array
    input_dict = {
        "shape": [4, 4],
        "dtype": np.dtype('float32'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array
    input_dict = {
        "shape": [10],
        "dtype": np.dtype('int32'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D bool array
    input_dict = {
        "shape": [2, 3, 5],
        "dtype": np.dtype('bool'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array
    input_dict = {
        "shape": [1],
        "dtype": np.dtype('float64'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D uint32 array
    input_dict = {
        "shape": [8, 16],
        "dtype": np.dtype('uint32'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D int16 array
    input_dict = {
        "shape": [3, 3, 3, 3],
        "dtype": np.dtype('int16'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D complex64 array
    input_dict = {
        "shape": [100],
        "dtype": np.dtype('complex64'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with an empty dimension
    input_dict = {
        "shape": [0, 10],
        "dtype": np.dtype('float32'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int8 array
    input_dict = {
        "shape": [5, 5],
        "dtype": np.dtype('int8'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 0-dimensional array
    input_dict = {
        "shape": [],
        "dtype": np.dtype('float32'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.zeros_3"] = zeros_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_3'.")


check_valid('jax.numpy.zeros', generated_inputs['jax.numpy.zeros_3'], lib="jax", suffix=3)
