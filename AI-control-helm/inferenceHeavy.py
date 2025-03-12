import numpy as np 
def inference_data(raw,model):
    print("Running Heavy AI")
    input_data = np.array([raw], dtype=np.float32)
    
    # Perform inference
    output_data = model.predict(input_data)
    
    node_data = output_data[0][0] 
    prob_rain = output_data[1][0] 
    
    return node_data, prob_rain