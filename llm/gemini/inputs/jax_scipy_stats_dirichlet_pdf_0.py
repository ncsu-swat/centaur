
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dirichlet_pdf_inputs():
    list_of_inputs = []

    # 1. K=3, x shape (3,), float32
    alpha = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # 2. K=3, x shape (2,), float32 (one entry fewer)
    alpha = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.2, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # 3. K=4, x shape (4,), float64
    alpha = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # 4. K=4, x shape (3,), float64
    alpha = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # 5. K=2, x shape (2,), float32
    alpha = np.array([2.0, 5.0], dtype=np.float32)
    x = np.array([0.3, 0.7], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # 6. K=2, x shape (1,), float32
    alpha = np.array([2.0, 5.0], dtype=np.float32)
    x = np.array([0.3], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # 7. K=5, x shape (5,), float32
    alpha = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # 8. K=5, x shape (4,), float32
    alpha = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    x = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # 9. K=10, x shape (10,), float64
    alpha = np.linspace(1.0, 5.0, 10, dtype=np.float64)
    x = np.ones(10, dtype=np.float64) / 10.0
    list_of_inputs.append({"x": x, "alpha": alpha})

    # 10. K=8, x shape (7,), float32
    alpha = np.linspace(0.5, 4.0, 8, dtype=np.float32)
    x = np.ones(7, dtype=np.float32) / 10.0
    list_of_inputs.append({"x": x, "alpha": alpha})

    return list_of_inputs

generated_inputs["jax.scipy.stats.dirichlet.pdf"] = dirichlet_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.dirichlet.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.dirichlet.pdf'.")


check_valid('jax.scipy.stats.dirichlet.pdf', generated_inputs['jax.scipy.stats.dirichlet.pdf'], lib="jax", suffix=0)
