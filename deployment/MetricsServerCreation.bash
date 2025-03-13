echo "Installing Metrics Server"
kubectl config use-context k3d-control
kubectl create configmap metrics-python --from-file=../Metrics-Server/test.py
kubectl create configmap ui-cm --from-file=../Metrics-Server/index.html

highestport=$(kubectl get svc -o wide | grep -Eo '500[0-9](\/|►)' | grep -Eo '[0-9]+' | sort -n | tail -1)
nexthighest=$((highestport + 1))
echo "Using port $nexthighest"
helm install metrics-server ../Metrics-Server/ \
  --set deployment.name=metrics-server \
  --set container.name=metrics-server \
  --set service.name=metrics-server \
  --set resultEndPoint=metrics-server \
  --set clusterName=metrics-server \
  --set service.port=$nexthighest \
  --set env[0].name=PORT \
  --set env[0].value=$nexthighest \