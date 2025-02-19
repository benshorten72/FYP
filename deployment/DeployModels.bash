#!/bin/bash
echo "Installing split model on edge"
kubectl create configmap $cluster_name-python --from-file=../AI-helm/test.py

helm install ai-model ../AI-helm \
  --set deployment.name=$cluster_name-ai \
  --set clusterName=$cluster_name \
  --set container.name=$cluster_name-ai \
  --set configmapName=$cluster_name-python \
  --set env[0].name=CLUSTER_NAME \
  --set env[0].value=$cluster_name \
  --set env[1].name=CLUSTER_RANK \
  --set env[1].value=$cluster_rank \

echo "Adding Device template to MQTT service. Will fail unnless both ingress and service are ready"
while true; do
    response=$(curl -4 -X POST -H "Content-Type: application/json" -d @./device-profiles/base-template-profile.json -s -o /dev/null -w "%{http_code}" http://$cluster_name.local/core-metadata/api/v3/deviceprofile)
    
    if [ "$response" -eq 201 ]; then
        echo "Received incorrect response : $response, Retrying..."

    elif [ "$response" -eq 409 ]; then
        echo "Profile already exists, not issue, continuing..."
        break
    elif [ "$response" -eq 207 ]; then
        echo "Profile already exists, not issue, continuing..."
        break
    else
        echo "Received response code: $response. Retrying loop"
    fi
    
    sleep 2
done
echo "Sensors can now communicate with cluster"

# echo "Installing split model on control"
# kubectl config use-context k3d-control

# helm install fy ../AI-helm/ \
#   --set deployment.name=$cluster_name \
#   --set container.name=$cluster_name \
#   --set service.name=$cluster_name \
#   --set resultEndPoint=$cluster_name \
#   --set clusterName=$cluster_name \
#   --set isCentralAI=True \
