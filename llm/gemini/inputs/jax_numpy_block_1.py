
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def block_inputs():
    list_of_inputs = []

    # Input 1: Standard 2D float32 array
    arrays = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({"arrays": arrays})

    # Input 2: 3D float32 array
    arrays = np.zeros((2, 4, 3), dtype=np.float32)
    list_of_inputs.append({"arrays": arrays})

    # Input 3: 4D float64 array
    arrays = np.ones((2, 2, 2, 2), dtype=np.float64)
    list_of_inputs.append({"arrays": arrays})

    # Input 4: 1D int32 array
    arrays = np.arange(12, dtype=np.int32)
    list_of_inputs.append({"arrays": arrays})

    # Input 5: 2D float32 array with negative values
    arrays = np.full((4, 5), -8.5, dtype=np.float32)
    list_of_inputs.append({"arrays": arrays})

    # Input 6: 2D int64 identity matrix
    arrays = np.eye(5, dtype=np.int64)
    list_of_inputs.append({"arrays": arrays})

    # Input 7: 3D int32 array with mixed positive/negative values
    arrays = np.random.randint(-50, 50, size=(2, 3, 2)).astype(np.int32)
    list_of_inputs.append({"arrays": arrays})

    # Input 8: Small 2D float32 array explicitly created
    arrays = np.array([[1.5, -2.5], [3.5, -4.5]], dtype=np.float32)
    list_of_inputs.append({"arrays": arrays})

    # Input 9: 2D uint8 array
    arrays = np.ones((6, 6), dtype=np.uint8)
    list_of_inputs.append({"arrays": arrays})

    # Input 10: 4D float32 array with singleton dimensions
    arrays = np.zeros((1, 5, 1, 5), dtype=np.float32)
    list_of_inputs.append({"arrays": arrays})

    return list_of_inputs

generated_inputs["jax.numpy.block_1"] = block_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.block_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.block_1'.")


check_valid('jax.numpy.block', generated_inputs['jax.numpy.block_1'], lib="jax", suffix=1)
