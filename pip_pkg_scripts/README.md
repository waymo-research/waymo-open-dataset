## Build a pip package locally using the docker container

```bash
docker build --tag=open_dataset_pip -f pip_pkg_scripts/build.Dockerfile .
mkdir /tmp/artifacts
docker run --mount type=bind,source=/tmp/artifacts,target=/tmp/artifacts -e "PYTHON_VERSION=3" -e "PYTHON_MINOR_VERSION=9" -e "PIP_MANYLINUX2010=1" -e "TF_VERSION=2.6.0" open_dataset_pip
```

This command will execute the `build.sh` inside the container, which clones the
github repository, builds the library and outputs `.whl` packages under
`/tmp/artifacts/`

To manually run commands inside docker

```bash
docker run -it --entrypoint='/bin/bash' --mount type=bind,source=/tmp/artifacts,target=/tmp/artifacts -e "PYTHON_VERSION=3" -e "PYTHON_MINOR_VERSION=9" -e "PIP_MANYLINUX2010=1" -e "TF_VERSION=2.6.0" open_dataset_pip
```

## Build a pip package locally without docker container

First follow quick start to install all the depdencies such as bazel. Then run

```bash
export PYTHON_VERSION=3
export PYTHON_MINOR_VERSION=9
export PIP_MANYLINUX2010=0
# Only support 2.4.0, 2.5.0, 2.6.0
export TF_VERSION=2.6.0
./pip_pkg_scripts/build.sh
```
