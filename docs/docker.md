Build the docker container:

```bash
docker build --tag=open_dataset .
```

Start a Jupyter notebook with `waymo_open_dataset` as a dependency:

```bash
docker run -p 8888:8888 open_dataset
```

Open http://0.0.0.0:8888 in browser and click on the `tutorial_local.ipynb` to
open the tutorial.
