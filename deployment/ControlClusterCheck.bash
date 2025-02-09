#!/bin/bash
export cluster_name="control"
k3d_cluster_name="k3d-"+$cluster_name
if kubectl config get-clusters | grep -q "^k3d-$cluster_name$"; then
    echo "Control Cluster '$cluster_name' exists in the kubeconfig."
else
    echo "Control Cluster '$cluster_name' does not exist in the kubeconfig."
    echo "Creating Control Cluster"
    ./CreateClusterBase.bash
fi