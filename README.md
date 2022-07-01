# GSoC_PlanetPrediction
## Kaggle Notebook Used for this task is - [Kaggle Notebook](https://www.kaggle.com/code/akshat5302/ai-in-space-0-00023-loss-100-acc/edit)
- After creating model I saved and used .h5 file of the model to create a Flask app using that .h5 file
- All files related to Flask App had been uploaded on Github repo (GSoC_PlanetPrediction)<br />
  Flask App can be accessed on local host at this IP- `127.0.0.1:5000`
## Application Containerization
- Then I containerized my Flask App and created its image using DockerFile and deployed it on DockerHub<br /> 
  Cmd to push the image - `docker push akshat5302/planet_pred:tagname`
## Deploying My Flask App on k3s cluster of Rancher
- For this Step I Created 2 Files `deployment.yaml` and `service.yaml`
- And runned them using cmd - `kubectl apply -f deployment.yaml` and `kubectl apply -f service.yaml` on k3s cluster
- Now I'm able to access my app inside KVM using `NodeIP:Nodeport`
- For the next phase I Added ` 172.16.230.162 mcm.rancher.aiic.suse` on /etc/hosts/ file of my local machine 
## Creating Namespace 
- After  logging into `mcm.rancher.aiic.suse` I created a namespace there
- And deployed my yaml files on that namespace.
- After that my Flask App is accessible on - `KVMIP:5000`
