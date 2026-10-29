
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
from jax.sharding import Mesh, PartitionSpec
import numpy as np

mesh = Mesh(np.array(jax.devices()), ("x",))
_mesh_ctx = jax.set_mesh(mesh)
_mesh_ctx.__enter__()


def empty_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    input_dict = {
        "shape": [8],
        "dtype": np.dtype("float32"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32
    input_dict = {
        "shape": [4, 6],
        "dtype": np.dtype("int32"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D bool
    input_dict = {
        "shape": [3, 5],
        "dtype": np.dtype("bool"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64
    input_dict = {
        "shape": [2, 3, 4],
        "dtype": np.dtype("float64"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D uint8
    input_dict = {
        "shape": [10],
        "dtype": np.dtype("uint8"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D int64
    input_dict = {
        "shape": [2, 2, 3, 2],
        "dtype": np.dtype("int64"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex64
    input_dict = {
        "shape": [5, 5],
        "dtype": np.dtype("complex64"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int16
    input_dict = {
        "shape": [1, 4, 8],
        "dtype": np.dtype("int16"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float16
    input_dict = {
        "shape": [1, 2, 1, 3, 2],
        "dtype": np.dtype("float16"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D uint32
    input_dict = {
        "shape": [12, 2],
        "dtype": np.dtype("uint32"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D complex128
    input_dict = {
        "shape": [3, 3, 3],
        "dtype": np.dtype("complex128"),
        "out_sharding": PartitionSpec(),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.empty_3"] = empty_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_3'.")


check_valid('jax.numpy.empty', generated_inputs['jax.numpy.empty_3'], lib="jax", suffix=3)
