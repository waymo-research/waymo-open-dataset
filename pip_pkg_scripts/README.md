# PIP package build scripts

Build a pip package locally using the docker container:

```
docker build --tag=open_dataset_pip -f pip_pkg_scripts/build.Dockerfile .
mkdir /tmp/pip_pkg_build
docker run --mount type=bind,source=/tmp/pip_pkg_build,target=/tmp/pip_pkg_build open_dataset_pip
```
This command will execute the `build.sh` inside the container, which clones the
github repository, builds the library and outputs `.whl` packages under
`/tmp/pip_pkg_build/`

Here is an example how to build a pip package for Python 2.7:

```
docker run --mount type=bind,source=/tmp/pip_pkg_build,target=/tmp/pip_pkg_build open_dataset_pip master 2
```

notice how arguments `master` and `2` are passed to the `build.sh` script to
initialize `GITHUB_BRANCH` and `PYTHON_VERSION` variables.
