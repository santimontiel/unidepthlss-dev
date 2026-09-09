USER_NAME := $(shell whoami)
IMAGE_NAME := unidepthlss
TAG_NAME := v1
CONTAINER_NAME := $(IMAGE_NAME)_container
GPU_ID := 0

UID := $(shell id -u)
GID := $(shell id -g)

HF_TOKEN := $(shell echo $$HF_TOKEN)
WANDB_API_KEY := $(shell echo $$WANDB_API_KEY)
# One dataset. Named to match every sibling repo in this family (tinycar-dev, fiery-radar, ...)
# so a machine that already trains those needs no new export.
NUSCENES_DATA_ROOT := $(shell echo $$NUSCENES_DATA_ROOT)

define run_docker
	@docker run -it --rm \
		--runtime nvidia \
		--net host \
		--gpus '"device=$(GPU_ID)"' \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(CONTAINER_NAME) \
		-u $(USER_NAME) \
		-v ./:/workspace \
		-v $(NUSCENES_DATA_ROOT):/data/nuscenes \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e TERM=xterm-256color \
		$(IMAGE_NAME):$(TAG_NAME) \
		/bin/bash -c $(1)
endef

check-env:
ifndef NUSCENES_DATA_ROOT
	$(error NUSCENES_DATA_ROOT is undefined. Please run 'export NUSCENES_DATA_ROOT=/your/path' first)
endif
	@if [ ! -d "$(NUSCENES_DATA_ROOT)" ]; then \
		echo "Error: NUSCENES_DATA_ROOT directory does not exist at $(NUSCENES_DATA_ROOT)"; \
		exit 1; \
	fi

.PHONY: build run attach jupyter clear
build:
	docker build deploy/docker -t $(IMAGE_NAME):$(TAG_NAME) --build-arg USER=$(USER_NAME) --build-arg UID=$(UID) --build-arg GID=$(GID)
	@echo "\nBuild complete!"
	@echo "Run 'make run' to start the container."

run: check-env
	$(call run_docker, "source deploy/docker/entrypoint.sh && bash")

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash -c bash

# Kept from the template: notebooks/visualize.ipynb is the surviving notebook (qualitative
# BEV figures). Clear its outputs before saving -- they are megabytes of images.
jupyter: check-env
	$(call run_docker, "jupyter notebook")

clear:
	@rm -rf .cache/
	@rm -rf .venv/
	@rm -rf unidepthlss.egg-info/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .ipynb_checkpoints/
	@echo "Cleaned up the project directory."

