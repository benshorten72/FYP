#!/bin/bash
# Not to be done with control cluster
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
    read -p "Enter the cluster name (alphanumeric, dashes, or underscores only): " cluster_name
    
    # Validate the input
    if validate_input "$cluster_name"; then
      break
    else
      echo "Invalid cluster name. Please use only alphanumeric characters, dashes, or underscores, and avoid spaces."
    fi
  done
fi
kubectl config use-context k3d-$cluster_name
echo "Uninstalling sensors"
for release in $(helm list --all-namespaces | awk '/^$cluster_name/ {print $1}'); do
  helm uninstall "$cluster_name"
done

echo "Deassociating it with control cluster"
curl -4 -X POST http://control.local/control/delete_cluster \
-H "Content-Type: application/json" \
-d "{\"name\": \"$cluster_name\""

echo "Deleting cluster"
k3d cluster delete $cluster_name
