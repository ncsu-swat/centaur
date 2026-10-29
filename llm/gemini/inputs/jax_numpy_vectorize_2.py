
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp
import numpy as np


class CallableTuple(tuple):

    def __call__(self, *args, **kwargs):
        return self[0](*args, **kwargs)


def vectorize_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, scalar elementwise sine
    input_dict = {
        "pyfunc": CallableTuple((lambda x: jnp.sin(x),)),
        "excluded": [],
        "signature": "()->()",
        "args": np.array([0.0, 0.5, 1.0, -1.5, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, vector reduction (sum)
    input_dict = {
        "pyfunc": CallableTuple((lambda x: jnp.sum(x),)),
        "excluded": [],
        "signature": "(n)->()",
        "args": np.random.randn(4, 5).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, scalar elementwise polynomial
    input_dict = {
        "pyfunc": CallableTuple((lambda x: x**2 - 2.0 * x + 1.0,)),
        "excluded": [],
        "signature": "()->()",
        "args": np.random.randn(3, 3).astype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, vector max along last dimension
    input_dict = {
        "pyfunc": CallableTuple((lambda x: jnp.max(x),)),
        "excluded": [],
        "signature": "(k)->()",
        "args": np.random.randn(2, 3, 4).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int32 array, vector reverse
    input_dict = {
        "pyfunc": CallableTuple((lambda x: x[::-1],)),
        "excluded": [],
        "signature": "(n)->(n)",
        "args": np.arange(12, dtype=np.int32).reshape(3, 4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, matrix trace reduction
    input_dict = {
        "pyfunc": CallableTuple((lambda x: jnp.trace(x),)),
        "excluded": [],
        "signature": "(n,n)->()",
        "args": np.random.randn(5, 3, 3).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, matrix transpose
    input_dict = {
        "pyfunc": CallableTuple((lambda x: x.T,)),
        "excluded": [],
        "signature": "(m,n)->(n,m)",
        "args": np.random.randn(4, 2, 3).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int64 array, scalar linear transformation with negative values
    input_dict = {
        "pyfunc": CallableTuple((lambda x: x * 3 + 7,)),
        "excluded": [],
        "signature": "()->()",
        "args": np.array([-10, -5, 0, 5, 10], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array, 1D vector mean reduction
    input_dict = {
        "pyfunc": CallableTuple((lambda x: jnp.mean(x),)),
        "excluded": [],
        "signature": "(d)->()",
        "args": np.random.randn(2, 2, 3, 4).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array, vector self-outer-product
    input_dict = {
        "pyfunc": CallableTuple((lambda x: jnp.outer(x, x),)),
        "excluded": [],
        "signature": "(n)->(n,n)",
        "args": np.random.randn(6, 3).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D bool array, scalar boolean negation
    input_dict = {
        "pyfunc": CallableTuple((lambda x: ~x,)),
        "excluded": [],
        "signature": "()->()",
        "args": np.array([True, False, True, True, False]),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.vectorize_2"] = vectorize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vectorize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vectorize_2'.")


check_valid('jax.numpy.vectorize', generated_inputs['jax.numpy.vectorize_2'], lib="jax", suffix=2)
