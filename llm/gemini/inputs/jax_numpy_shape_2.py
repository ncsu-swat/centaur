
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def shape_inputs():
    list_of_inputs = []

    # Input 1: Zero as np.int64
    list_of_inputs.append({"a": np.int64(0)})

    # Input 2: Positive small integer as np.int32
    list_of_inputs.append({"a": np.int32(1)})

    # Input 3: Negative small integer as np.int64
    list_of_inputs.append({"a": np.int64(-1)})

    # Input 4: Larger positive integer as np.int32
    list_of_inputs.append({"a": np.int32(42)})

    # Input 5: Larger negative integer as np.int64
    list_of_inputs.append({"a": np.int64(-100)})

    # Input 6: Large positive integer as np.int32
    list_of_inputs.append({"a": np.int32(100000)})

    # Input 7: Large negative integer as np.int64
    list_of_inputs.append({"a": np.int64(-100000)})

    # Input 8: Integer near 32-bit limit as np.int64
    list_of_inputs.append({"a": np.int64(2147483647)})

    # Input 9: Negative integer near 32-bit limit as np.int64
    list_of_inputs.append({"a": np.int64(-2147483648)})

    # Input 10: Value ten as np.int32
    list_of_inputs.append({"a": np.int32(10)})

    return list_of_inputs

generated_inputs["jax.numpy.shape_2"] = shape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.shape_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.shape_2'.")


check_valid('jax.numpy.shape', generated_inputs['jax.numpy.shape_2'], lib="jax", suffix=2)
