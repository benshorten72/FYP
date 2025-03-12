#!/bin/bash
export cluster_name="control"
export cluster_rank=0
k3d_cluster_name="k3d-"+$cluster_name
if kubectl config get-clusters | grep -q "^k3d-$cluster_name$"; then
    echo "Control Cluster '$cluster_name' exists in the kubeconfig."
else
    echo "Control Cluster '$cluster_name' does not exist in the kubeconfig."
    echo "Creating Control Cluster"
    while true; do
        echo "Is federated learning required for this testbed (y/n)?"
        read -p "" federated_check
        if [[ "$federated_check" =~ ^[Yy]$ ]]; then
            echo "Federated learning enabled"
            break
        elif [[ "$federated_check" =~ ^[Nn]$ ]]; then
            echo "Federated learning disabled"
            break
        else
            echo "Invalid input. Please enter 'y' or 'n'."
        fi
    done

    ./CreateClusterBase.bash
    kubectl config use-context k3d-control 
    echo "Deploying control container"
    kubectl create configmap control-python --from-file=../control-container/test.py
    helm install control ../control-container/

    if [[ "$federated_check" =~ ^[Yy]$ ]]; then
            ./ParameterServerCreation.bash
    fi
    ./MetricsServerCreation.bash
fi
