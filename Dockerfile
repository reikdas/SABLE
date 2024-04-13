FROM ubuntu:22.04

# install the necessary packages
RUN apt-get -y update && apt-get -y install \
    cmake \
    gcc \
    g++ \
    python3 \
    git \
    wget \
    libomp-dev \
    parallel 

# install Intel oneMKL
RUN wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/2f3a5785-1c41-4f65-a2f9-ddf9e0db3ea0/l_onemkl_p_2024.1.0.695_offline.sh && chmod +x l_onemkl_p_2024.1.0.695_offline.sh && sh ./l_onemkl_p_2024.1.0.695_offline.sh --silent -a --eula accept

# add intel oneMKL to the path
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2024.1.0/lib/intel64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/intel/oneapi/mkl/2024.1.0/lib/intel64:$LIBRARY_PATH
ENV CPATH=/opt/intel/oneapi/mkl/2024.1.0/include:$CPATH

# create new user called sableuser
RUN useradd -ms /bin/bash sableuser

# copy the SABLE source code to the container directory /home/sableuser/SABLE
COPY . /home/sableuser/SABLE
RUN chown -R sableuser /home/sableuser/

USER sableuser

# set the working directory to /home/sableuser/SABLE
WORKDIR /home/sableuser/SABLE
