#!/bin/bash
DH_USERNAME="santimontiel"
IMAGE_NAME="unidepthlss"
TAG_NAME="v1"
DOCKER_UID=$(id -u)
DOCKER_GID=$(id -g)

set -e

docker build ../docker -t ${IMAGE_NAME}:${TAG_NAME} --build-arg USER=${USER} --build-arg UID=${DOCKER_UID} --build-arg GID=${DOCKER_GID}
docker tag ${IMAGE_NAME}:${TAG_NAME} ${DH_USERNAME}/${IMAGE_NAME}:${TAG_NAME}
docker push ${DH_USERNAME}/${IMAGE_NAME}:${TAG_NAME}
echo "Docker image pushed to DockerHub as ${DH_USERNAME}/${IMAGE_NAME}:${TAG_NAME}"