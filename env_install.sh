#!/usr/bin/env bash


# install anaconda3.
#cd ~/
#wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
#bash Anaconda3-2019.07-Linux-x86_64.sh

# make sure your annaconda3 is added to bashrc
source ~/.bashrc

conda create --name second
conda activate second
conda install -y pytorch=1.4.0 torchvision cudatoolkit=10.0 python=3.6.8 Pillow==6.1 numpy=1.16 -c pytorch # make sure python=3.6.8
pip install numba scikit-image scipy matplotlib fire tensorboardX protobuf opencv-python psutil flask flask-cors

cd ..
git clone git@github.com:traveller59/spconv.git --recursive
sudo apt-get install libboostall-dev
python setup.py bdist_wheel
cd ./dist
pip install
