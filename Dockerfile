FROM ubuntu:22.04

# install the necessary packages
RUN apt-get -y update && apt-get -y install \
    cmake \
    gcc \
    g++ \
    python3 \
    wget \
    libomp-dev \
    parallel \
    gpg-agent

# install Intel oneMKL
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get -y update && apt-get -y install intel-oneapi-mkl-devel

# add intel oneMKL to the path
ENV INTEL_DIR=/opt/intel
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:$LIBRARY_PATH
ENV CPATH=/opt/intel/oneapi/mkl/latest/include:$CPATH

# create new user called sableuser
RUN useradd -ms /bin/bash sableuser

WORKDIR /home/sableuser/
COPY . /home/sableuser/SABLE
WORKDIR /home/sableuser/SABLE
ENV SABLE_ROOT_DIR=/home/sableuser/SABLE
RUN chown -R sableuser /home/sableuser/

USER sableuser

RUN python3 -m pip install -r requirements.txt
