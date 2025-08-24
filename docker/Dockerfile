ARG PYTHON_VERSION=3.11.8
ARG LINUX_DISTRO=bookworm

ARG BUILD_IMAGE=python:${PYTHON_VERSION}-${LINUX_DISTRO}

FROM ${BUILD_IMAGE}

LABEL maintainer=taehyung.yu@vuno.co
LABEL org.opencontainers.image.source=https://github.com/vuno/pr-review-agent

RUN apt-get -y clean && \
    apt-get -y update

ARG TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get install -y tzdata && \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

COPY --link requirements.txt /opt/tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /opt/tmp/requirements.txt

