#!/bin/bash
./ControlClusterCheck.bash
source ./CreateClusterBase.bash
helm install edgex ../edgex-helm/
SERVICE_NAME="app-load-balancer"
EXTERNAL_IP=$(kubectl get svc $SERVICE_NAME -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "External IP assigned: $EXTERNAL_IP"
echo "Ingress ready on http://$cluster_name.local"
./DeployModels.bash

echo "Registering device and rank with control cluster"
curl -4 -X POST http://control.local/control/add_cluster \
-H "Content-Type: application/json" \
-d "{\"name\": \"$cluster_name\", \"rank\": $cluster_rank}"
