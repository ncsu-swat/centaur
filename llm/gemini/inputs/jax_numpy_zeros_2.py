
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
from jax.sharding import Mesh, PartitionSpec
import numpy as np

_mesh = Mesh(np.array(jax.devices()), ("x",))
jax.set_mesh(_mesh).__enter__()


def zeros_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar, float32
    input_dict = {
        "shape": (),
        "dtype": np.dtype("float32"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-D array, int32
    input_dict = {
        "shape": (5,),
        "dtype": np.dtype("int32"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1-D array, bool
    input_dict = {
        "shape": (8,),
        "dtype": np.dtype("bool"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2-D array, float32
    input_dict = {
        "shape": (3, 4),
        "dtype": np.dtype("float32"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2-D array, uint8
    input_dict = {
        "shape": (6, 2),
        "dtype": np.dtype("uint8"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1-D empty array (zero size), float32
    input_dict = {
        "shape": (0,),
        "dtype": np.dtype("float32"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3-D array with a zero dimension, complex64
    input_dict = {
        "shape": (2, 0, 4),
        "dtype": np.dtype("complex64"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3-D array, int32
    input_dict = {
        "shape": (2, 3, 5),
        "dtype": np.dtype("int32"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4-D array, float16
    input_dict = {
        "shape": (1, 2, 3, 4),
        "dtype": np.dtype("float16"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5-D array, int16
    input_dict = {
        "shape": (2, 2, 2, 2, 2),
        "dtype": np.dtype("int16"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2-D array, int8
    input_dict = {
        "shape": (4, 8),
        "dtype": np.dtype("int8"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 1-D array, uint32
    input_dict = {
        "shape": (16,),
        "dtype": np.dtype("uint32"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.zeros_2"] = zeros_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_2'.")


check_valid('jax.numpy.zeros', generated_inputs['jax.numpy.zeros_2'], lib="jax", suffix=2)
