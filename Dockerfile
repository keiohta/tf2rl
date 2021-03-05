# syntax = docker/dockerfile:experimental
FROM python:3.7

RUN apt update \
	&& apt install -y --no-install-recommends \
	python-opengl \
	&& apt clean \
	&& rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
	pip install -U \
	cpprb \
	joblib \
	matplotlib \
	scipy \
	tensorflow==2.2.* \
	tensorflow_probability==0.10.*

COPY setup.py /tf2rl/setup.py
COPY tf2rl /tf2rl/tf2rl

RUN pip install /tf2rl tensorflow_probability==0.10.* && rm -rf /tf2rl


CMD ["/bin/bash"]
