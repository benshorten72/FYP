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
# Prompt the user for the cluster name
while true; do
  read -p "Enter the cluster name (alphanumeric, dashes, or underscores only): " cluster_name
  
  # Validate the input
  if validate_input "$cluster_name"; then
    break
  else
    echo "Invalid cluster name. Please use only alphanumeric characters, dashes, or underscores, and avoid spaces."
  fi
done

#!/bin/bash

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

# Output the provided range
echo "You have defined the IP range as 172.100.150.$ip_start_range-172.100.150.$ip_end_range"

echo "Creating docker subnet 172.100.0.0/16 for testbed"
docker network create --subnet 172.100.0.0/16 testbed

echo "Using k3d to create cluster - disabling inbuilt loadbalancer and using testbed network"
k3d cluster create $cluster_name --k3s-arg "--disable=servicelb@server:0" --no-lb --wait --network testbed

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
helm install ingress-nginx ingress-nginx/ -n ingress-nginx --create-namespace --set clusterName=$cluster_name

INGRESS_LB_IP=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Add $INGRESS_LB_IP to /etc/hosts"

echo "$INGRESS_LB_IP $cluster_name.local" | sudo tee -a /etc/hosts

helm install edgex edgex-helm/
SERVICE_NAME="app-load-balancer"
TIMEOUT=300  # Timeout in seconds (5 minutes)
INTERVAL=5   # Check interval in seconds

echo "Waiting for external IP to be assigned to service: $SERVICE_NAME..."

end=$((SECONDS+$TIMEOUT))
while [ $SECONDS -lt $end ]; do
  EXTERNAL_IP=$(kubectl get svc $SERVICE_NAME -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  if [ -n "$EXTERNAL_IP" ]; then
    echo "External IP assigned: $EXTERNAL_IP"
    exit 0
  fi
  echo "Waiting for external IP... (Retrying in $INTERVAL seconds)"
  sleep $INTERVAL
done

echo "Timeout reached. External IP not assigned to service: $SERVICE_NAME"
exit 1


echo "Ingress ready on http://$cluster_name.local"
