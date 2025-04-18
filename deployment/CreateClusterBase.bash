#!/bin/bash
echo "Creating Edge Cluster..."
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
    read -p "Enter the cluster name (alphanumeric, dashes, or underscores only): " cluster_name
    
    # Validate the input
    if validate_input "$cluster_name"; then
      break
    else
      echo "Invalid cluster name. Please use only alphanumeric characters, dashes, or underscores, and avoid spaces."
    fi
  done
fi
export cluster_name
echo "Cluster name: $cluster_name"
# Loop to get the starting IP range
while true; do
  read -p "Enter the cluster external starting IP range (e.g., 1,2,3...): " ip_start_range
  if [[ "$ip_start_range" =~ ^[0-9]+$ ]]; then
    break
  else
    echo "Invalid input. Please enter a numeric value."
  fi
done

# Loop to get the ending IP range
while true; do
  read -p "Enter the cluster external ending IP range (e.g., 4,5,6...): " ip_end_range
  if [[ "$ip_end_range" =~ ^[0-9]+$ ]] && [[ $ip_end_range -gt $ip_start_range ]]; then
    break
  else
    echo "Invalid input. Please enter a numeric value greater than the starting IP range."
  fi
done

if [[ -z "${cluster_rank}" ]]; then
    read -p "Enter the cluster rank for scheduling: " cluster_rank
fi
export cluster_rank
# Output the provided range
echo "You have defined the IP range as 172.100.150.$ip_start_range-172.100.150.$ip_end_range"

echo "Creating docker subnet 172.100.0.0/16 for testbed"
docker network create --subnet 172.100.0.0/16 testbed
echo "Cluster name: $cluster_name"
# Loop to get the starting IP range
while true; do
  read -p "Enter the cluster memory resource limit (in GB): " memory_resource_limit
  if [[ "$memory_resource_limit" =~ ^[0-64]+$ ]]; then
    break
  else
    echo "Invalid input. Please enter a numeric value."
  fi
done
while true; do
  read -p "Enter the cluster core resource limit (how many cores to use): " cpu_resource_limit
  if [[ "$cpu_resource_limit" =~ ^[0-64]+$ ]]; then
    break
  else
    echo "Invalid input. Please enter a numeric value."
  fi
done
if [[ $cluster_name != "control" ]]; then
  while true; do
      echo "Does this need to preform split learning? This will require a more computationally expensive \n running on edge due to need for Tensorflow and a full python image rather than a silm \n
      (y/n)?"
      read -p "" response
      if [[ "$response" =~ ^[Yy]$ ]]; then
          echo "Split learning enabled"
          export split_check="True"
          export image="pidgeot72/control-ai"
          export inference_script="inferenceHeavy.py"
          export model_file_name="model.keras"
          break
      elif [[ "$response" =~ ^[Nn]$ ]]; then
          echo "Split learning disabled"
          export split_check="False"
          export image="pidgeot72/resource-constrainted-ai"
          export inference_script="inferenceLite.py"
          export model_file_name="model.tflite"
          break
      else
          echo "Invalid input. Please enter 'y' or 'n'."
      fi
  done
fi

if [[ $cluster_name != "control" ]]; then
  while true; do
      echo "Does the control model need early stopping to improve training (y/n)?"
      read -p "" choice
      if [[ "$choice" =~ ^[Yy]$ ]]; then
          echo "Split learning enabled"
          export USE_EARLY_STOPPING="True"
          break
      elif [[ "$choice" =~ ^[Nn]$ ]]; then
          export USE_EARLY_STOPPING="False"
          break
      else
          echo "Invalid input. Please enter 'y' or 'n'."
      fi
  done
fi

echo "Using k3d to create cluster - disabling inbuilt loadbalancer and using testbed network"
# use correct model depending on cluster type
if [[ $cluster_name == "control" ]]; then
    model_abs_path=$(realpath ./models/trained_control_model.keras)
  elif [[ $split_check == "True" ]]; then
    model_abs_path=$(realpath ./models/trained_edge_model.keras)
  elif [[ $split_check == "False" ]]; then
    model_abs_path=$(realpath ./models/trained_edge_model.tflite)
fi
echo "Using $model_abs_path, because split learning enabled = $split_check "
k3d cluster create $cluster_name --k3s-arg "--disable=servicelb@server:0" --no-lb --wait --network testbed --k3s-arg "--disable=traefik"  --volume $model_abs_path:/mnt/model \
  --agents-memory 2G

kubectl config use-context k3d-$cluster_name

helm repo add metallb https://metallb.github.io/metallb
helm install metallb -n metallb --create-namespace metallb/metallb
echo "Waiting for metallb to deploy..."
# DO not take out sleep, it breaks otherwise
sleep 10
kubectl wait -n metallb -l app.kubernetes.io/component=controller --for=condition=ready pod --timeout=120s
kubectl wait -n metallb -l app.kubernetes.io/component=speaker --for=condition=ready pod --timeout=120s

kubectl apply -f - <<EOF
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: ip-pool
  namespace: metallb
spec:
  addresses:
  - 172.100.150.${ip_start_range}-172.100.150.${ip_end_range}
EOF

kubectl apply -f - <<EOF
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: advertisement-l2
  namespace: metallb
spec:
  ipAddressPools:
  - ip-pool
EOF

# Install the ingress-nginx Helm chart
kubectl edit validatingwebhookconfiguration ingress-nginx-admission
helm install ingress-nginx ../ingress-nginx/ -n ingress-nginx --create-namespace --set clusterName=$cluster_name --set controller.admissionWebhooks.enabled=false
INGRESS_LB_IP=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Add $INGRESS_LB_IP to /etc/hosts"

echo "$INGRESS_LB_IP $cluster_name.local" | sudo tee -a /etc/hosts
EXTERNAL_IP=$(kubectl get svc $SERVICE_NAME -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "External IP assigned: $EXTERNAL_IP"
echo "Ingress ready on http://$cluster_name.local"