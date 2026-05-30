
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def shape_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float as np.float64
    list_of_inputs.append({"a": np.float64(3.14)})

    # Input 2: Negative float as np.float32
    list_of_inputs.append({"a": np.float32(-2.718)})

    # Input 3: Zero float as np.float64
    list_of_inputs.append({"a": np.float64(0.0)})

    # Input 4: Negative zero float as np.float32
    list_of_inputs.append({"a": np.float32(-0.0)})

    # Input 5: Large float as np.float64
    list_of_inputs.append({"a": np.float64(1.23e10)})

    # Input 6: Small float as np.float32
    list_of_inputs.append({"a": np.float32(4.56e-10)})

    # Input 7: Positive infinity as np.float64
    list_of_inputs.append({"a": np.float64(float('inf'))})

    # Input 8: Negative infinity as np.float32
    list_of_inputs.append({"a": np.float32(float('-inf'))})

    # Input 9: NaN as np.float64
    list_of_inputs.append({"a": np.float64(float('nan'))})

    # Input 10: Float as np.float16
    list_of_inputs.append({"a": np.float16(0.0001)})

    # Input 11: Specific value as np.float64
    list_of_inputs.append({"a": np.float64(-999.999)})

    return list_of_inputs

generated_inputs["jax.numpy.shape_3"] = shape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.shape_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.shape_3'.")


check_valid('jax.numpy.shape', generated_inputs['jax.numpy.shape_3'], lib="jax", suffix=3)
