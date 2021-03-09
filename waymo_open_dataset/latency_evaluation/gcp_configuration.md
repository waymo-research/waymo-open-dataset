# Latency evaluation server configuration

## General GCP setup

Enable Compute Engine API if needed (will ask once on first load of GCE)

## Create VM

# Settings

 - Use 'Marketplace'
 - Search for `Deep Learning VM`
 - Confirm hardware:
   - name: `wod-latency-evaluator-N` (N = 1...)
   - zone: `us-west-1b`
   - `n1-standard-8` (8 vCPUs, 30 GB memory)
   - 1 x `NVIDIA Tesla V100`
   - Install NVIDIA GPU driver automatically on first startup
   - Boot disk type: SSD persistent disk, 200GB

### Post creation

Edit the instance, scroll to `Service Account`, then to `Access scopes`
* Storage: `Read Write` (required for submission loading and storing results).
* Cloud Source Repositories: 'Read Only' (required for operations on docker
  images).

If you are getting permission errors when writing files, double check storage
permissions, as bucket permissions alone are not sufficient.

### Install docker

```bash
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

### Verify docker

```bash
sudo docker run hello-world
```

### Install Nvidia runtime

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo groupadd docker
sudo usermod -aG docker ${USER}
```

### Verify Nvidia runtime

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Configure Docker helper

```bash
gcloud auth configure-docker us-west1-docker.pkg.dev
```
