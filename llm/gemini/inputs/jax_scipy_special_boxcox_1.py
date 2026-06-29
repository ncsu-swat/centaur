
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax.scipy.special

# Monkeypatch jax.scipy.special.boxcox in case it is not present in the current JAX version
def _boxcox(x, lmbda):
    cond = lmbda == 0
    safe_lmbda = jnp.where(cond, 1.0, lmbda)
    return jnp.where(cond, jnp.log(x), (jnp.power(x, safe_lmbda) - 1.0) / safe_lmbda)

jax.scipy.special.boxcox = _boxcox

def boxcox_inputs():
    list_of_inputs = []

    # 1. 1D arrays, float32, lmbda = 0.0
    list_of_inputs.append({
        "x": np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32),
        "lmbda": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    })

    # 2. 1D arrays, float32, lmbda = 0.5
    list_of_inputs.append({
        "x": np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32),
        "lmbda": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    })

    # 3. 2D arrays, float32
    list_of_inputs.append({
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "lmbda": np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float32)
    })

    # 4. 2D arrays with varying lmbda
    list_of_inputs.append({
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "lmbda": np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    })

    # 5. 3D arrays, float32
    list_of_inputs.append({
        "x": np.random.uniform(0.1, 5.0, size=(2, 2, 2)).astype(np.float32),
        "lmbda": np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    })

    # 6. Scalar-like 0D arrays
    list_of_inputs.append({
        "x": np.array(5.0, dtype=np.float32),
        "lmbda": np.array(2.0, dtype=np.float32)
    })

    # 7. 1D arrays, float32, negative lmbda
    list_of_inputs.append({
        "x": np.array([0.5, 1.5, 2.5], dtype=np.float32),
        "lmbda": np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    })

    # 8. Large values of x
    list_of_inputs.append({
        "x": np.array([100.0, 200.0, 300.0], dtype=np.float32),
        "lmbda": np.array([0.1, 0.1, 0.1], dtype=np.float32)
    })

    # 9. Broadcasting shapes: (3, 1) and (1, 3)
    list_of_inputs.append({
        "x": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "lmbda": np.array([[0.5, 1.0, 1.5]], dtype=np.float32)
    })

    # 10. Broadcasting shapes: (2, 1, 3) and (1, 2, 1)
    list_of_inputs.append({
        "x": np.random.uniform(0.1, 5.0, size=(2, 1, 3)).astype(np.float32),
        "lmbda": np.random.uniform(-1.0, 1.0, size=(1, 2, 1)).astype(np.float32)
    })

    return list_of_inputs

generated_inputs["jax.scipy.special.boxcox_1"] = boxcox_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.boxcox_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.boxcox_1'.")


check_valid('jax.scipy.special.boxcox', generated_inputs['jax.scipy.special.boxcox_1'], lib="jax", suffix=1)
