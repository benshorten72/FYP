#!/bin/bash
echo "Installing split model on edge"
while true; do
    echo "Does this need to preform split learning? This will require a more computationally expensive \n running on edge due to need for Tensorflow and a full python image rather than a silm \n
    (y/n)?"
    read -p "" response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Split learning enabled"
        inference_script="inferenceHeavy.py"
        break
    elif [[ "$response" =~ ^[Nn]$ ]]; then
        echo "Split learning disabled"
        inference_script="inferenceLite.py"

        break
    else
        echo "Invalid input. Please enter 'y' or 'n'."
    fi
done

kubectl create configmap $cluster_name-python --from-file=../AI-helm/test.py
kubectl create configmap inference --from-file=../AI-helm/$inference_script

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

echo "Installing AI model on control"
kubectl config use-context k3d-control
kubectl create configmap $cluster_name-ai --from-file=../AI-control-helm/test.py

highestport=$(kubectl get svc -o wide | grep -Eo '500[0-9](\/|►)' | grep -Eo '[0-9]+' | sort -n | tail -1)
nexthighest=$((highestport + 1))
echo "Using port $nexthighest"
helm install $cluster_name-ai ../AI-control-helm/ \
  --set deployment.name=$cluster_name \
  --set container.name=$cluster_name \
  --set service.name=$cluster_name \
  --set resultEndPoint=$cluster_name \
  --set clusterName=$cluster_name \
  --set service.port=$nexthighest \
  --set env[0].name=PORT \
  --set env[0].value=$nexthighest \


echo "Now accesible on http://control.local/$cluster_name"