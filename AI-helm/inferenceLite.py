# 2 models
from ai_edge_litert.interpreter import Interpreter
import numpy as np

def inference_data(raw,interpreter):
    print("Running light AI")
    input_details = interpreter.get_input_details()
    input_data = np.array([raw], dtype=np.float32)
    # Set input tensor
    input_index = input_details[0]['index']
    interpreter.set_tensor(input_index, input_data)
    # GET THE OUTPUT DATA
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    rain_data = interpreter.get_tensor(output_details[0]['index'])
    node_data = interpreter.get_tensor(output_details[1]['index'])
    node_data = node_data[0] #Send this to control
    prob_rain = rain_data[0]
    return node_data, prob_rain