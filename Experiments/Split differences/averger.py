import csv

def calculate_average_of_second_column(csv_file_path):
    try:
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            column_values = []
            
            for row in reader:
                if len(row) >= 2: 
                    try:
                        value = float(row[1])
                        column_values.append(value)
                    except ValueError:
                        print(f"Skipping non-numeric value: {row[1]}")
            
            if not column_values:
                print("No valid numeric data found in the second column.")
                return None
            
            average = sum(column_values) / len(column_values)
            return average
    
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None

if __name__ == "__main__":
    csv_path = "./S6/memory/edge_python_memory_usage.txt"

    result = calculate_average_of_second_column(csv_path)
    
    if result is not None:
        print(f"Average of the second column: {result:.2f}")

#----------------
# S2 - control=755.44 edge=498.47
# S3 - control=748.45 edge=503.74
# S4 - control=744.23 edge=509.71
# S5 - control=735.23 edge=522.45
# S6 - control=722.70 edge=533.92
#----------------

# Learning comparative
# MSE
# naive control=22.7 control learning period taken out of=0.19 edge=11834.27
# partial control= .27 control learning period taken out of=0.38 edge= 41.98 
# trained control= 0.14 control learning period taken out of=0.19 edge= 5.77


#----------------
# Split analysis

# S2 - control=0.10 edge=3.70
# S3 - control= 0.10 edge=3.86
# S4 - control=0.11 edge=0.11
# S5 - control=0.17 edge=0.19
# S6 - control=0.12 edge=0.10

# MSE 
# S2 - control=0.07 edge=31
# S3 - control=0.12 edge=28.07
# S4 - control=0.13 edge=14.2
# S5 - control=0.31 edge=1.2
# S6 - control=0.29 edge=0.29
#----------------
