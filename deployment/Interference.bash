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
validate_number() {
  [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 0 ]
}
if [[ -z "${cluster_name}" ]]; then
  while true; do
    echo -e "Enter the edge cluster name you would like add interference to (alphanumeric, dashes, or underscores only):"
    read -p "" cluster_name
    
    # Validate the input
    if (validate_input "$cluster_name") && (kubectl config get-clusters | grep -q "^k3d-$cluster_name$"); then
      break
      else
      echo "Invalid cluster name."
    fi
done
fi
while true; do
  echo -e "Enter network delay in milliseconds (e.g., 100):"
  read -p "" delay_ms
  if validate_number "$delay_ms"; then
    break
  else
    echo "Invalid number. Please enter a positive integer."
  fi
done
while true; do
  echo -e "Enter packet loss percentage (0-100):"
  read -p "" loss_percent
  if validate_number "$loss_percent" && [ "$loss_percent" -ge 0 ] && [ "$loss_percent" -le 100 ]; then
    break
  else
    echo "Invalid percentage. Please enter a number between 0 and 100."
  fi
done
cat <<EOF | kubectl --context "k3d-$cluster_name" apply -f -
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: external-interferer
spec:
  selector:
    matchLabels:
      app: external-interferer
  template:
    metadata:
      labels:
        app: external-interferer
    spec:
      hostNetwork: true
      containers:
      - name: interferer
        image: docker.io/nicolaka/netshoot
        command: ["/bin/sh", "-c"]
        args:
          - |
            echo "Starting network interference with ${delay_ms}ms delay and ${loss_percent}% packet loss"
            INTERFACE=\$(ip -o route show default | awk '{print \$5}')
            tc qdisc del dev \$INTERFACE root 2>/dev/null || true
            tc qdisc add dev \$INTERFACE root netem delay ${delay_ms}ms loss ${loss_percent}%
            sleep infinity
        securityContext:
          privileged: true
          capabilities:
            add: ["NET_ADMIN"]
      tolerations:
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule
EOF

echo "Network interference applied to cluster 'k3d-$cluster_name' with:"
echo "- Delay: ${delay_ms}ms"
echo "- Packet loss: ${loss_percent}%"

