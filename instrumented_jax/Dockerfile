# Use Ubuntu 22.04 as a base image for modern toolchains
FROM ubuntu:jammy

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies, Python, and LLVM/Clang
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    cpanminus \
    git curl bubblewrap patch make unzip bzip2 opam libgmp-dev python2.7 \
    software-properties-common \
    patchelf \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    clang \
    llvm \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    software-properties-common \
    ninja-build \
    clang \
    llvm-dev \
    libomp-dev \
    libopenblas-dev \
    libblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libssl-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libboost-all-dev \
    lld \
    lcov \
    tmux \
    btop \
    htop \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

ENV CC=clang
ENV CXX=clang++
ENV LDFLAGS="-fuse-ld=lld"

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install Bazel — JAX 0.4.35 requires Bazel 6.x same as TF
ARG BAZEL_VERSION=6.5.0
RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    rm bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

# Set working directory
WORKDIR /app

# Clone JAX source code
# NOTE: Unlike TF, JAX uses build/build.py as a Bazel wrapper — no ./configure needed
# XLA source path inside container: /app/jax/bazel-jax/external/xla/xla
# (NOT external/local_xla/xla like TF — this affects xla_source_dir in coverage scripts)
ARG JAX_VERSION=v0.4.35
RUN git clone --depth 1 --branch ${JAX_VERSION} https://github.com/google/jax.git

WORKDIR /app/jax

# Install JAX build dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy scipy wheel packaging requests auditwheel

ENV CC=clang
ENV CXX=clang++

# Build jaxlib with coverage instrumentation flags
# KEY DIFFERENCE FROM TF: JAX uses build/build.py which wraps Bazel internally
# --bazel_options passes flags through to the underlying Bazel invocation
# No ./configure step, no direct bazel build call
RUN python build/build.py \
    --bazel_options="--action_env=CC=clang" \
    --bazel_options="--action_env=CXX=clang++" \
    --bazel_options="--copt=-fprofile-instr-generate" \
    --bazel_options="--copt=-fcoverage-mapping" \
    --bazel_options="--linkopt=-fprofile-instr-generate" \
    --bazel_options="--linkopt=-fcoverage-mapping" \
    --bazel_options="--collect_code_coverage"

# Install the instrumented jaxlib wheel
# No auditwheel repair needed — build.py already produces a repaired wheel
RUN pip install dist/jaxlib-*.whl

# Install JAX pure Python package separately
RUN pip install "jax==${JAX_VERSION#v}"

# Verify _jaxlib.so path — needed for coverage binary detection in docker_coverage.py
RUN python -c "import jaxlib; import os; print('jaxlib path:', os.path.dirname(jaxlib.__file__))"

# Copy the centaur repository into the Docker image
COPY . /workspace/repo

WORKDIR /workspace/repo

# Copy instrumented jaxlib wheel to repo directory for reference
RUN mkdir -p /workspace/repo/instrumented_jax && \
    cp /app/jax/dist/jaxlib-*.whl /workspace/repo/instrumented_jax/

CMD ["bash"]