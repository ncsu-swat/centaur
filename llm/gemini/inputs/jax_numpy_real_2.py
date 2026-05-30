
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def real_inputs():
    list_of_inputs = []

    # Input 1: Basic positive Python integer
    list_of_inputs.append({"val": 5})

    # Input 2: Negative Python integer
    list_of_inputs.append({"val": -10})

    # Input 3: Zero
    list_of_inputs.append({"val": 0})

    # Input 4: Large Python integer
    list_of_inputs.append({"val": 1000000})

    # Input 5: numpy int32 positive
    list_of_inputs.append({"val": np.int32(42)})

    # Input 6: numpy int32 negative
    list_of_inputs.append({"val": np.int32(-100)})

    # Input 7: numpy int64
    list_of_inputs.append({"val": np.int64(9876543210)})

    # Input 8: numpy uint8
    list_of_inputs.append({"val": np.uint8(255)})

    # Input 9: numpy int16 negative
    list_of_inputs.append({"val": np.int16(-32768)})

    # Input 10: numpy int_ scalar
    list_of_inputs.append({"val": np.int_(15)})

    return list_of_inputs

generated_inputs["jax.numpy.real_2"] = real_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.real_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.real_2'.")


check_valid('jax.numpy.real', generated_inputs['jax.numpy.real_2'], lib="jax", suffix=2)
