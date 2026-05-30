
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
from jax.sharding import PartitionSpec, Mesh
import copy

# Set up a global mesh context so PartitionSpec can be canonicalized
devices = np.array(jax.devices())
mesh = Mesh(devices, ('x',))
mesh_ctx = jax.set_mesh(mesh)
mesh_ctx.__enter__()

def zeros_inputs():
    list_of_inputs = []

    shapes = [10, 5, 100, 1, 1000, 2, 8, 50, 256, 12]
    dtypes = [
        np.float32, 
        np.int32, 
        np.float64, 
        np.bool_, 
        np.int64,
        np.complex64, 
        np.uint8, 
        np.int16, 
        np.uint32, 
        np.int8
    ]

    for i in range(10):
        # Use empty PartitionSpec() to avoid numpy min/max errors in the test framework
        input_dict = {
            "shape": shapes[i],
            "dtype": dtypes[i],
            "out_sharding": PartitionSpec()
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
