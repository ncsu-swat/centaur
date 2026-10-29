
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec

try:
    _mesh = Mesh(np.array(jax.devices()), ("x",))
    if hasattr(jax, "set_mesh"):
        _mesh_context = jax.set_mesh(_mesh)
        _mesh_context.__enter__()
    else:
        _mesh.__enter__()
except Exception:
    pass


def empty_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    input_dict = {
        "shape": [4],
        "dtype": "float32",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array
    input_dict = {
        "shape": [2, 3],
        "dtype": "int32",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D bool array with size 0
    input_dict = {
        "shape": [0],
        "dtype": "bool",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float16 array
    input_dict = {
        "shape": [8, 8],
        "dtype": "float16",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D bfloat16 array
    input_dict = {
        "shape": [1, 4, 2],
        "dtype": "bfloat16",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array
    input_dict = {
        "shape": [64],
        "dtype": "int64",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D scalar complex64 array
    input_dict = {
        "shape": [],
        "dtype": "complex64",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D uint8 array
    input_dict = {
        "shape": [2, 2, 2, 2],
        "dtype": "uint8",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 array with zero dimension
    input_dict = {
        "shape": [2, 0, 4],
        "dtype": "float64",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int16 array
    input_dict = {
        "shape": [16, 32],
        "dtype": "int16",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D uint32 array
    input_dict = {
        "shape": [3, 5],
        "dtype": "uint32",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D int8 array
    input_dict = {
        "shape": [2, 3, 4, 5],
        "dtype": "int8",
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.empty_6"] = empty_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_6'.")


check_valid('jax.numpy.empty', generated_inputs['jax.numpy.empty_6'], lib="jax", suffix=6)
