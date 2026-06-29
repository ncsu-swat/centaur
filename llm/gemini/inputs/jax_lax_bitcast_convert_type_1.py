
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitcast_convert_type_inputs():
    list_of_inputs = []

    # 1. float32 to int32 (same size)
    list_of_inputs.append({
        "operand": np.array([1.0, 2.0], dtype=np.float32),
        "new_dtype": np.int32
    })

    # 2. int64 to float64 (same size)
    list_of_inputs.append({
        "operand": np.array([1, 2], dtype=np.int64),
        "new_dtype": np.float64
    })

    # 3. uint8 to int8 (same size)
    list_of_inputs.append({
        "operand": np.array([0, 127], dtype=np.uint8),
        "new_dtype": np.int8
    })

    # 4. int16 to uint16 (same size)
    list_of_inputs.append({
        "operand": np.array([-1, 1], dtype=np.int16),
        "new_dtype": np.uint16
    })

    # 5. float32 to uint8 (downcast size)
    list_of_inputs.append({
        "operand": np.array([1.0], dtype=np.float32),
        "new_dtype": np.uint8
    })

    # 6. int64 to int32 (downcast size)
    list_of_inputs.append({
        "operand": np.array([1], dtype=np.int64),
        "new_dtype": np.int32
    })

    # 7. float64 to int16 (downcast size)
    list_of_inputs.append({
        "operand": np.array([1.0], dtype=np.float64),
        "new_dtype": np.int16
    })

    # 8. int32 to int8 (downcast size)
    list_of_inputs.append({
        "operand": np.array([1], dtype=np.int32),
        "new_dtype": np.int8
    })

    # 9. int32 to int64 (upcast size, last dim = 2)
    list_of_inputs.append({
        "operand": np.array([1, 2], dtype=np.int32),
        "new_dtype": np.int64
    })

    # 10. uint8 to float32 (upcast size, last dim = 4)
    list_of_inputs.append({
        "operand": np.array([0, 0, 128, 63], dtype=np.uint8),
        "new_dtype": np.float32
    })

    return list_of_inputs

generated_inputs["jax.lax.bitcast_convert_type_1"] = bitcast_convert_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitcast_convert_type_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitcast_convert_type_1'.")


check_valid('jax.lax.bitcast_convert_type', generated_inputs['jax.lax.bitcast_convert_type_1'], lib="jax", suffix=1)
