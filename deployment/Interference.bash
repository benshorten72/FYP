helm repo add chaos-mesh https://charts.chaos-mesh.org
helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh --create-namespace \
  --set controllerManager.enableClusterMode=true \
  --set controllerManager.service.type=LoadBalancer
