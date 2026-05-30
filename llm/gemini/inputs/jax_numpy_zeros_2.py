
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
from jax.sharding import PartitionSpec

def zeros_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (4,),
        "dtype": np.dtype('float32'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (2, 3),
        "dtype": np.dtype('int32'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (8, 8, 8),
        "dtype": np.dtype('float64'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D boolean array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (10, 5),
        "dtype": np.dtype('bool'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D complex64 array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (1, 100),
        "dtype": np.dtype('complex64'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D uint32 array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (16, 16),
        "dtype": np.dtype('uint32'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float16 array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (3, 3, 3, 3),
        "dtype": np.dtype('float16'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int64 array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (5,),
        "dtype": np.dtype('int64'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D uint8 array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (2, 4, 8),
        "dtype": np.dtype('uint8'),
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int16 array, empty PartitionSpec (replicated)
    input_dict = {
        "shape": (12, 12),
        "dtype": np.dtype('int16'),
        "out_sharding": PartitionSpec()
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
