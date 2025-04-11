import random
from datetime import datetime, timedelta

def randomize_data(input_file, output_file):
    """
    Replaces all data columns (except 'date') with random values.
    
    Args:
        input_file (str): Path to input CSV/file with original data
        output_file (str): Path to save randomized data
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(',')
            
            # Keep first column (date) as-is
            date_part = parts[0]
            
            # Generate random values for other columns
            random_values = [f"{random.uniform(0, 100):.3f}" for _ in parts[1:]]
            
            # Recombine with original date
            new_line = f"{date_part},{','.join(random_values)}\n"
            outfile.write(new_line)

# Example Usage
randomize_data('./data/weather_data_config.csv', './data/randomized_data.csv')