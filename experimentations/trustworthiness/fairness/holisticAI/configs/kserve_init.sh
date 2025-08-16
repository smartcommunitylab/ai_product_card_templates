kind create cluster
kubectl config get-contexts
kubectl config use-context kind-kind

# local deployment of kserve
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.15/hack/quick_install.sh" | bash

kubectl create namespace kserve-test
# check Inferenceservice status
kubectl get inferenceservices sklearn-hiring -n kserve-test

# apply the InferenceSefvice to the cluster
kubectl apply -n kserve-test -f kserve.yaml

# figure out the address the istio is listening on
kubectl get svc istio-ingressgateway -n istio-system

# Some useful commands for debugging 
kubectl get namespaces
kubectl get pods -n <namespace>
kubectl -n <namespace> logs <pod-name>
kubectl delete -f sklearn-wine.yaml -n mlflow-kserve-test