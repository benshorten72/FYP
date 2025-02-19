#!/bin/bash

validate_input() {
  local input="$1"
  # Check if input is empty, contains spaces, or special characters
  if [[ -z "$input" || "$input" =~ [^a-zA-Z0-9_-] ]]; then
    return 1 
  else
    return 0  
  fi
}

if [[ -z "${cluster_name}" ]]; then
  while true; do
    echo -e "Enter the edge cluster name you would like datasource to \ncommunicate with (alphanumeric, dashes, or underscores only):"
    read -p "" cluster_name
    
    # Validate the input
    if (validate_input "$cluster_name") && (kubectl config get-clusters | grep -q "^k3d-$cluster_name$"); then
      break
      else
      echo "Invalid cluster name."
    fi
done
fi
export cluster_name
if [[ -z "${filename}" ]]; then

    echo -e "Enter csv file in located in data. File must have data names in top row"
    read -r filename
fi
export filename

# Define the directory where the file should be located
directory="./data"

# Check if the file exists
if [[ ! -f "$directory/$filename" ]]; then
  echo "Error: File '$filename' does not exist in the '$directory' directory."
  exit 1
fi

# Check if the file has a .csv extension
if [[ "$filename" != *.csv ]]; then
  echo "Error: File '$filename' is not a CSV file."
  exit 1
fi

header=$(head -n 1 $directory/$filename)
IFS=','
header_array=($header)
# Iterate through each item in the array
for column in "${header_array[@]}"; do
    export column
    ./SensorCreation.bash

done