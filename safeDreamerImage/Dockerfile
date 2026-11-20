# The open-source part of the SafeDreamer image: Ubuntu 20.04, Python 3.8, the pinned
# packages, the simulator and SafeDreamer. Built on GitHub's servers, where the build has
# the admin rights the cluster does not give us.


FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG SAFEDREAMER_COMMIT=529221f02b54d5093e304d877eb2aaf6da63a9ff
ARG SAFETY_GYMNASIUM_COMMIT=064c0c3f1dac523ffdf1e4daddba8262779344d4

# Ubuntu 20.04's python3 is Python 3.8, the version SafeDreamer was written for.
# build-essential and python3-dev are there in case a package has to be compiled.
# The lib* packages are for drawing camera images without a screen:
# libegl1/libgl1/libopengl0/libglvnd0 for the GPU route, libosmesa6 for the CPU fallback.
# libglib2.0-0 is needed by OpenCV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential git ca-certificates \
        libegl1 libgl1 libopengl0 libglvnd0 libosmesa6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Apptainer's `--nv` brings the node's NVIDIA graphics library into the container at run
# time, but not this small file that tells the system the library exists.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY requirements.txt /opt/requirements.txt
RUN python3 -m pip install --no-cache-dir pip==24.0 setuptools==69.5.1 wheel==0.43.0 \
    && python3 -m pip install --no-cache-dir -r /opt/requirements.txt

# The simulator. Installed in "editable" mode so it runs from this clone:
# a normal install would leave out some of its 3D asset files.
# --no-deps because requirements.txt already pins what it needs.
RUN git clone https://github.com/PKU-Alignment/safety-gymnasium.git /opt/safety-gymnasium \
    && git -C /opt/safety-gymnasium checkout "$SAFETY_GYMNASIUM_COMMIT" \
    && python3 -m pip install --no-cache-dir --no-deps -e /opt/safety-gymnasium

# SafeDreamer is run as scripts from its folder, so it is cloned and not installed.
RUN git clone https://github.com/PKU-Alignment/SafeDreamer.git /opt/SafeDreamer \
    && git -C /opt/SafeDreamer checkout "$SAFEDREAMER_COMMIT"

# Draw camera images on the GPU. Override with MUJOCO_GL=osmesa to use the CPU.
ENV MUJOCO_GL=egl
# Apptainer mounts the home folder; this stops Python picking up packages from ~/.local.
ENV PYTHONNOUSERSITE=1
ENV PYTHONUNBUFFERED=1
ENV LC_ALL=C.UTF-8

LABEL safedreamer.commit=$SAFEDREAMER_COMMIT
LABEL safety_gymnasium.commit=$SAFETY_GYMNASIUM_COMMIT

# Fail the build if any of the main packages does not import.
RUN python3 -c "import jax, mujoco, safety_gymnasium, tensorflow, tensorflow_probability; print('imports ok: jax', jax.__version__, 'mujoco', mujoco.__version__)"
