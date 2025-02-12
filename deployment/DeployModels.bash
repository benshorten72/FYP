#!/bin/bash
echo "Installing split model on edge"
cluster_name="xcx"
kubectl create configmap $cluster_name-python --from-file=../AI-helm/test.py

helm install ai-model ../AI-helm \
  --set deployment.name=$cluster_name-ai \
  --set container.name=$cluster_name-ai \
  --set configmapName=$cluster_name-python \
  --set env[0].name=CLUSTER_NAME \
  --set env[0].value=$cluster_name \



# echo "Installing split model on control"
# kubectl config use-context k3d-control

# helm install fy ../AI-helm/ \
#   --set deployment.name=$cluster_name \
#   --set container.name=$cluster_name \
#   --set service.name=$cluster_name \
#   --set resultEndPoint=$cluster_name \
#   --set clusterName=$cluster_name \
#   --set isCentralAI=True \
