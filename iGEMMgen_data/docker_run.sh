#!/bin/bash
# Get the folders
PWD=`pwd`
DOCKER_DIR=/`basename $PWD`
GPU=$1
DECK=$2
CTNRNAME=ConvContainer$GPU

# IMAGE="rocm/tensorflow-private:rocm3.5-tf1.15-horovod-resnet50_v1.5-dev"
IMAGE="rocm/dev-ubuntu-18.04:latest"

docker run --name $CTNRNAME -it -d  -e "GPU="$1 --rm --network=host --device=/dev/kfd \
--device=/dev/dri/renderD128 \
--device=/dev/dri/renderD129 \
--device=/dev/dri/renderD130 \
--device=/dev/dri/renderD131 \
--device=/dev/dri/renderD132 \
--device=/dev/dri/renderD133 \
--device=/dev/dri/renderD134 \
--device=/dev/dri/renderD135 \
--ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --user $(id -u):$(id -g) -w $DOCKER_DIR -v=`pwd`:$DOCKER_DIR  $IMAGE

docker exec -t $CTNRNAME $DOCKER_DIR/conv$DECK.sh
docker stop $CTNRNAME 



