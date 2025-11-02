
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: Basic string content
    filename = np.array("test_file.txt", dtype=np.string_)
    contents = np.array("Hello World!", dtype=np.string_)
    input_dict = {
        "name": "test_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: Empty string content
    filename = np.array("empty_file.txt", dtype=np.string_)
    contents = np.array("", dtype=np.string_)
    input_dict = {
        "name": "empty_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: Multi-line string content
    filename = np.array("multiline_file.txt", dtype=np.string_)
    contents = np.array("Line 1\nLine 2\nLine 3", dtype=np.string_)
    input_dict = {
        "name": "multiline_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: String with special characters
    filename = np.array("special_chars.txt", dtype=np.string_)
    contents = np.array("Hello!\n@#$%^&*()", dtype=np.string_)
    input_dict = {
        "name": "special_chars.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: Numeric string content
    filename = np.array("numeric_file.txt", dtype=np.string_)
    contents = np.array("123456789", dtype=np.string_)
    input_dict = {
        "name": "numeric_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: Binary content
    filename = np.array("binary_file.bin", dtype=np.string_)
    contents = np.array(b"\x00\x01\x02\x03", dtype=np.string_)
    input_dict = {
        "name": "binary_file.bin",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Long string content
    filename = np.array("long_file.txt", dtype=np.string_)
    contents = np.array("A" * 1000, dtype=np.string_)
    input_dict = {
        "name": "long_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: String with spaces and tabs
    filename = np.array("spaces_file.txt", dtype=np.string_)
    contents = np.array("   \t\t\t   ", dtype=np.string_)
    input_dict = {
        "name": "spaces_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: String with newline and carriage return
    filename = np.array("cr_newline_file.txt", dtype=np.string_)
    contents = np.array("\r\nHello\nWorld\r", dtype=np.string_)
    input_dict = {
        "name": "cr_newline_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: String with ASCII characters only
    filename = np.array("ascii_file.txt", dtype=np.string_)
    contents = np.array("Hello World! This is a test.", dtype=np.string_)
    input_dict = {
        "name": "ascii_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.WriteFile"] = generate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.WriteFile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.WriteFile'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.WriteFile', generated_inputs['tf.raw_ops.WriteFile'], lib="tf", suffix=0)
