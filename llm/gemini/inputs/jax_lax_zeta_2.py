
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def zeta_inputs():
    list_of_inputs = []

    # Input 1: Standard values (Riemann zeta function equivalent, q=1.0)
    input_dict = {"x": 2.0, "q": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-integer x > 1, integer q
    input_dict = {"x": 2.5, "q": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-integer q in (0, 1)
    input_dict = {"x": 3.0, "q": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive x
    input_dict = {"x": 10.0, "q": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: q > 1 with integer x
    input_dict = {"x": 4.0, "q": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Non-integer x and q
    input_dict = {"x": 1.5, "q": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small q > 0
    input_dict = {"x": 2.0, "q": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large q
    input_dict = {"x": 3.5, "q": 15.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x close to 1
    input_dict = {"x": 1.05, "q": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher order integer x and q
    input_dict = {"x": 6.0, "q": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Fractional values for both
    input_dict = {"x": 4.25, "q": 1.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.zeta_2"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.zeta_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.zeta_2'.")


check_valid('jax.lax.zeta', generated_inputs['jax.lax.zeta_2'], lib="jax", suffix=2)
