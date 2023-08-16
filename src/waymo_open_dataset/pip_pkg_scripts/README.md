## Build a pip package locally using the docker container

```bash
mkdir /tmp/wod
cd src
docker build \
    --tag=open_dataset_pip\
    -f waymo_open_dataset/waymo_open_dataset/pip_pkg_scripts/build.Dockerfile\
    --build-arg USERNAME=$USER\
    --build-arg USER_UID=$`(id -u `$USER) .
docker run --mount type=bind,source=/tmp/wod,target=/tmp/wod open_dataset_pip
```

This command will execute the `build.sh` inside the container, which runs all
tests and builds the waymo_open_dataset wheel package for the lib. The outputs
 `.whl` will be under `/tmp/wod/` folder.

To manually run commands inside docker

```bash
docker run -it --entrypoint='/bin/bash' --mount type=bind,source=$PWD,target=/tmp/repo  open_dataset_pip
cd src
bazelisk build //waymo_open_dataset/waymo_open_dataset/pip_pkg_scripts:wheel_manylinux
```


## Build a pip package locally without docker container

Follow the
[instructions to install bazelisk](https://bazel.build/install/bazelisk) and run

```bash
cd src
bazelisk build //waymo_open_dataset/waymo_open_dataset/pip_pkg_scripts:wheel
```

Output wheel build for you host platform will in
`/tmp/repo/bazel-bin/waymo_open_dataset/pip_pkg_scripts`

## Update requirements

Known dependencies are "pinned" in `../requirements.in`. To update the full list
of transitive dependencies in `requirements.txt` execute:

```
cd src
bazelisk run //waymo_open_dataset:requirements.update
```

## Reporting an issue

To quickly diagnose any issue please attach a log file created by the
following bazel command:

```
bazelisk test ... --test_output=errors --subcommands --verbose_failures \
  --sandbox_debug   --keep_going 2>&1 | tee bazel_wod_test.log
```


