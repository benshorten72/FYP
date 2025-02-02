#!/bin/bash

# validate_input() {
#   local input="$1"
#   # Check if input is empty, contains spaces, or special characters
#   if [[ -z "$input" || "$input" =~ [^a-zA-Z0-9_-] ]]; then
#     return 1 
#   else
#     return 0  
#   fi
# }
# # Prompt the user for the cluster name
# while true; do
#   read -p "Enter a device name (alphanumeric, dashes, or underscores only): " device_name
  
#   # Validate the input
#   if validate_input "$cluster_name"; then
#     break
#   else
#     echo "Invalid device name. Please use only alphanumeric characters, dashes, or underscores, and avoid spaces."
#   fi
# done

curl -X POST http://wow.local/api/v3/device \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MyVirtualTemperatureDevice",
    "profileName": "Temperature",
    "serviceName": "device-virtual",
    "adminState": "UNLOCKED",
    "protocols": {
      "example": {
        "address": "localhost",
        "port": "49990"
      }
    }
  }'