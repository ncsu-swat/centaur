
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def generate_jax_numpy_can_cast_inputs():
    list_of_inputs = []

    # Input 1: safe cast from int32 to int64
    list_of_inputs.append({
        "from_": "int32",
        "to": "int64",
        "casting": "safe"
    })

    # Input 2: safe cast from float64 to complex128
    list_of_inputs.append({
        "from_": "float64",
        "to": "complex128",
        "casting": "safe"
    })

    # Input 3: unsafe cast from complex128 to float64
    list_of_inputs.append({
        "from_": "complex128",
        "to": "float64",
        "casting": "unsafe"
    })

    # Input 4: same_kind cast from int8 to float32
    list_of_inputs.append({
        "from_": "int8",
        "to": "float32",
        "casting": "same_kind"
    })

    # Input 5: unsafe cast from float32 to int32
    list_of_inputs.append({
        "from_": "float32",
        "to": "int32",
        "casting": "unsafe"
    })

    # Input 6: safe cast from uint8 to int16
    list_of_inputs.append({
        "from_": "uint8",
        "to": "int16",
        "casting": "safe"
    })

    # Input 7: no casting allowed, identical types
    list_of_inputs.append({
        "from_": "int32",
        "to": "int32",
        "casting": "no"
    })

    # Input 8: safe cast from float16 to float32
    list_of_inputs.append({
        "from_": "float16",
        "to": "float32",
        "casting": "safe"
    })

    # Input 9: safe cast from bool to int32
    list_of_inputs.append({
        "from_": "bool",
        "to": "int32",
        "casting": "safe"
    })

    # Input 10: same_kind cast from int64 to float64
    list_of_inputs.append({
        "from_": "int64",
        "to": "float64",
        "casting": "same_kind"
    })

    # Input 11: same_kind cast from float32 to float16
    list_of_inputs.append({
        "from_": "float32",
        "to": "float16",
        "casting": "same_kind"
    })

    return list_of_inputs

generated_inputs["jax.numpy.can_cast_4"] = generate_jax_numpy_can_cast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.can_cast_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.can_cast_4'.")


check_valid('jax.numpy.can_cast', generated_inputs['jax.numpy.can_cast_4'], lib="jax", suffix=4)
