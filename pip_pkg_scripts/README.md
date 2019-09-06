# PIP package build scripts

Build a pip package locally using the docker container:

```
docker build --tag=open_dataset_pip -f pip_pkg_scripts/build.Dockerfile .
mkdir /tmp/pip_pkg_build
docker run --mount type=bind,source=/tmp/pip_pkg_build,target=/tmp/pip_pkg_build -e "GITHUB_BRANCH=master" -e "PYTHON_VERSION=3" -e "PYTHON_MINOR_VERSION=5" -e "PIP_MANYLINUX2010=1" open_dataset_pip
```
This command will execute the `build.sh` inside the container, which clones the
github repository, builds the library and outputs `.whl` packages under
`/tmp/pip_pkg_build/`
