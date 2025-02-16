#!/bin/bash
# I need to delete the sensor on the control, the reference of it on the mqtt device service cluster
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
    echo -e "Enter the edge cluster name you would like to delete the datasource\n on(alphanumeric, dashes, or underscores only):"
    read -p "" cluster_name
    
    # Validate the input
    if (validate_input "$cluster_name") && (kubectl config get-clusters | grep -q "^k3d-$cluster_name$"); then
      break
      else
      echo "Invalid cluster name."
    fi
done
fi

kubectl config use-context k3d-control
echo "Enter column/data name sensor is sending"
read -p "" column
sanitized=$(echo "$column" | tr -cd 'a-z0-9.-' | sed -E 's/^[^a-z0-9]*//;s/[^a-z0-9]*$//;s/\.+/./g;s/^-*//;s/-*$//')
kubectl delete cm $cluster_name-$sanitized
# Ensure the sanitized string starts and ends with an alphanumeric character
sanitized=$(echo "$sanitized" | sed -E 's/^[^a-z0-9]*//;s/[^a-z0-9]*$//')
curl -X DELETE -H "Content-Type: application/json"  http://$cluster_name.local/core-metadata/api/v3/device/name/$column
helm uninstall $cluster_name-$sanitized