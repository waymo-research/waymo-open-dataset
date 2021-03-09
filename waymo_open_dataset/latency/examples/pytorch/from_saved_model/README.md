# Build instructions

Make sure that your `/etc/docker/daemon.json` file has `nvidia-container-runtime`
setup correctly, so that Docker can find the GPUs while building the image.
See: [SO answer](https://stackoverflow.com/a/61737404) for more details.
