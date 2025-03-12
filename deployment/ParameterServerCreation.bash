echo "Installing parameter server control"
kubectl config use-context k3d-control
kubectl create configmap parameter-python --from-file=../Parameter-Server/test.py
highestport=$(kubectl get svc -o wide | grep -Eo '500[0-9](\/|►)' | grep -Eo '[0-9]+' | sort -n | tail -1)
nexthighest=$((highestport + 1))
echo "Using port $nexthighest"
helm install parameter-server ../Parameter-Server/ \
  --set deployment.name=parameter-server \
  --set container.name=parameter-server \
  --set service.name=parameter-server \
  --set resultEndPoint=parameter-server \
  --set clusterName=parameter-server \
  --set service.port=$nexthighest \
  --set env[0].name=PORT \
  --set env[0].value=$nexthighest \