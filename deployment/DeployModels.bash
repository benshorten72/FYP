#!/bin/bash
echo "Installing split model on edge"
helm install ai-model ../AI-helm


echo "Installing split model on control"
kubectl config use-context k3d-control

helm install fy ../AI-helm/ \
  --set deployment.name=$cluster_name \
  --set container.name=$cluster_name \
  --set service.name=$cluster_name \
  --set resultEndPoint=$cluster_name \
  --set clusterName=$cluster_name \
  --set isCentralAI=True \

echo ""