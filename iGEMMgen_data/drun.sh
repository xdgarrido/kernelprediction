PWD=`pwd`
DOCKER_DIR=/`basename $PWD`
#export ROCR_VISIBLE_DEVICES=$1
CTNRNAME=GpuContainer$1
IMAGE="rocm/dev-ubuntu-18.04:latest"
#IMAGE="rocm/tensorflow-private:rocm3.5-tf1.15-horovod-resnet50_v1.5-dev"

docker run -it  --rm --name $CTNRNAME --network=host  --ipc=host --shm-size 16G  -v=`pwd`:$DOCKER_DIR \
-v /data:/data -w $DOCKER_DIR --privileged --rm --device=/dev/kfd \
--device=/dev/dri/renderD128 \
--device=/dev/dri/renderD129 \
--device=/dev/dri/renderD130 \
--device=/dev/dri/renderD131 \
--device=/dev/dri/renderD132 \
--device=/dev/dri/renderD133 \
--device=/dev/dri/renderD134 \
--device=/dev/dri/renderD135 \
--group-add video \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined $IMAGE 

