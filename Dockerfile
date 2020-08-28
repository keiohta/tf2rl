# syntax = docker/dockerfile:experimental
FROM python:3.7

RUN apt update \
	&& apt install -y --no-install-recommends \
	python-opengl \
	&& apt clean \
	&& rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
	pip install -U \
	matplotlib \
	tensorflow==2.2.* \
	tensorflow_probability==0.10.* \
	tf2rl

CMD ["/bin/bash"]
