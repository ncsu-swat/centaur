
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec

_mesh = Mesh(np.array(jax.devices()), ("x",))
if hasattr(jax, "set_mesh"):
    jax.set_mesh(_mesh).__enter__()
else:
    _mesh.__enter__()


def zeros_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar size
    input_dict = {
        "shape": 1,
        "dtype": np.dtype(np.float32),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, size 4
    input_dict = {
        "shape": 4,
        "dtype": np.dtype(np.int32),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, size 8
    input_dict = {
        "shape": 8,
        "dtype": np.dtype(np.float64),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: zero-sized array
    input_dict = {
        "shape": 0,
        "dtype": np.dtype(np.float32),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bool, size 2
    input_dict = {
        "shape": 2,
        "dtype": np.dtype(np.bool_),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64, size 16
    input_dict = {
        "shape": 16,
        "dtype": np.dtype(np.int64),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8, size 7
    input_dict = {
        "shape": 7,
        "dtype": np.dtype(np.uint8),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64, size 10
    input_dict = {
        "shape": 10,
        "dtype": np.dtype(np.complex64),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int16, size 32
    input_dict = {
        "shape": 32,
        "dtype": np.dtype(np.int16),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, size 64
    input_dict = {
        "shape": 64,
        "dtype": np.dtype(np.complex128),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: uint32, size 3
    input_dict = {
        "shape": 3,
        "dtype": np.dtype(np.uint32),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: int8, size 128
    input_dict = {
        "shape": 128,
        "dtype": np.dtype(np.int8),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.zeros_1"] = zeros_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_1'.")


check_valid('jax.numpy.zeros', generated_inputs['jax.numpy.zeros_1'], lib="jax", suffix=1)
