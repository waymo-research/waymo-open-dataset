Build the docker container:

```bash
docker build --tag=open_dataset .
```

Start a Jupyter notebook with `waymo_open_dataset` as a dependency:

```bash
docker run -p 8888:8888 open_dataset
```
