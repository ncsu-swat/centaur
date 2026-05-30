
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax
from jax.sharding import Mesh, PartitionSpec
import numpy as np
import copy

devices = jax.devices()
mesh = Mesh(np.array(devices, dtype=object), ('x',))
jax.set_mesh(mesh).__enter__()

def empty_inputs():
    list_of_inputs = []
    
    dtypes = [
        np.dtype('float32'),
        np.dtype('int32'),
        np.dtype('bool'),
        np.dtype('float64'),
        np.dtype('int64'),
        np.dtype('uint8'),
        np.dtype('float16'),
        np.dtype('int16'),
        np.dtype('complex64'),
        np.dtype('complex128')
    ]
    
    shapes = [5, 10, 1, 64, 128, 8, 32, 16, 3, 256]

    for s, d in zip(shapes, dtypes):
        out_sharding = PartitionSpec()
        
        input_dict = {
            "shape": s,
            "dtype": d,
            "out_sharding": out_sharding
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.empty_1"] = empty_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_1'.")


check_valid('jax.numpy.empty', generated_inputs['jax.numpy.empty_1'], lib="jax", suffix=1)
