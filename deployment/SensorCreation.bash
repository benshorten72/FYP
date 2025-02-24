#!/bin/bash
# I need to feed in a device profile json so the cluster
# can listen out for it.
#It also needs a csv file

# Values I need - Cluster name, filename, column 

validate_input() {
  local input="$1"
  # Check if input is empty, contains spaces, or special characters
  if [[ -z "$input" || "$input" =~ [^a-zA-Z0-9_-] ]]; then
    return 1 
  else
    return 0  
  fi
}

# Prompt the user for the cluster name
# Check if env var cluster name is already defined
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
kubectl config use-context k3d-$cluster_name
echo "Getting external IP for MQTT"
SERVICE_NAME="app-load-balancer"
EXTERNAL_IP=$(kubectl get svc $SERVICE_NAME -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "External IP assigned: $EXTERNAL_IP"
kubectl config use-context k3d-control
if [[ -z "${filename}" ]]; then

echo -e "Enter the name of datasource file located in datafile (CSV under 1MB only):"
read -r filename
fi
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

# If all checks pass, proceed with the file
echo "File '$filename' is valid and ready for processing."
echo "Enter column to read in csv file"
read -p "" column


sanitized=$(echo "$column" | tr -cd 'a-z0-9.-' | sed -E 's/^[^a-z0-9]*//;s/[^a-z0-9]*$//;s/\.+/./g;s/^-*//;s/-*$//')
# Ensure the sanitized string starts and ends with an alphanumeric character
sanitized=$(echo "$sanitized" | sed -E 's/^[^a-z0-9]*//;s/[^a-z0-9]*$//')
# Print the sanitized result
echo "Sanitized output: $sanitized"
file="$directory/$filename"

kubectl create configmap $cluster_name-$sanitized --from-file=$file
kubectl create configmap $cluster_name-$sanitized-python --from-file=../Sensor/test.py

helm install $cluster_name-$sanitized ../Sensor/ \
  --set columnName=$column \
  --set deployment.name=$cluster_name-$sanitized-sensor \
  --set container.name=$cluster_name-$sanitized-sensor \
  --set configmapName=$cluster_name-$sanitized \
  --set env[0].name=COLUMN_NAME \
  --set env[0].value=$column \
  --set env[1].name=CLUSTER_NAME \
  --set env[1].value=$cluster_name \
  --set env[2].name=FILE_NAME \
  --set env[2].value=$filename \
  --set env[3].name=MQTT_IP \
  --set env[3].value=$EXTERNAL_IP

# Define bash variables
HOST="$EXTERNAL_IP"
PORT="1883"
TOPIC="$cluster_name/$column"
COMMAND_TOPIC="test"

# Create the JSON string with embedded variables
JSON=$(cat <<EOF
[
  {
    "apiVersion": "v3",
    "device": {
      "name": "$column",
      "description": "A test sensor device",
      "adminState": "UNLOCKED",
      "operatingState": "UP",
      "protocols": {
        "mqtt": {
          "host": "$HOST",
          "port": "$PORT",
          "topic": "$TOPIC",
          "CommandTopic": "$COMMAND_TOPIC"
        }
      },
      "labels": ["MQTT", "sensor"],
      "profileName": "Generic-MQTT-String-Float-Device",
      "serviceName": "device-mqtt"
    }
  }
]
EOF
)

# Write the JSON string to a file
echo "Associating device with Generic-MQTT-String-Float-Device profile on EdgeX MQTT"
echo "$JSON" > device-profiles/generic-device.json
curl -4 -X POST -H "Content-Type: application/json" -d @./device-profiles/generic-device.json http://$cluster_name.local/core-metadata/api/v3/device
echo "Sensor deployed and assoicated with edge cluster MQTT device service"