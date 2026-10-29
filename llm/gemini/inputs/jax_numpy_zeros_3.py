
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
from jax.sharding import Mesh, PartitionSpec
import numpy as np

_mesh = Mesh(np.array(jax.devices()), ("data",))
_mesh_context = jax.set_mesh(_mesh)
_mesh_context.__enter__()


def zeros_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    input_dict = {
        "shape": [4],
        "dtype": np.dtype(np.float32),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array
    input_dict = {
        "shape": [2, 3],
        "dtype": np.dtype(np.int32),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array
    input_dict = {
        "shape": [64, 16],
        "dtype": np.dtype(np.float64),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D bool array
    input_dict = {
        "shape": [1, 5, 10],
        "dtype": np.dtype(np.bool_),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D scalar int64 array
    input_dict = {
        "shape": [],
        "dtype": np.dtype(np.int64),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex64 array
    input_dict = {
        "shape": [64],
        "dtype": np.dtype(np.complex64),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with zero dimension
    input_dict = {
        "shape": [0, 4],
        "dtype": np.dtype(np.float32),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D uint8 array
    input_dict = {
        "shape": [8, 16, 8],
        "dtype": np.dtype(np.uint8),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D int16 array
    input_dict = {
        "shape": [3, 3, 3, 3],
        "dtype": np.dtype(np.int16),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex128 array
    input_dict = {
        "shape": [8, 64],
        "dtype": np.dtype(np.complex128),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D float16 array
    input_dict = {
        "shape": [128],
        "dtype": np.dtype(np.float16),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 5D int8 array
    input_dict = {
        "shape": [1, 2, 1, 2, 1],
        "dtype": np.dtype(np.int8),
        "out_sharding": PartitionSpec(),
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
