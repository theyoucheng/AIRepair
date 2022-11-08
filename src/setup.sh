#install anaconda
echo "first installing anaconda"
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh --output anaconda.sh
bash anaconda.sh

echo "creating virtual environment for torch"
echo "DeepRepair ART"
conda create -n torch python
conda activate torch
conda install pytorch torchvision tqdm scipy scikit-learn matplotlib numpy
pip install onnx
pip install diffabs
conda deactivate

echo "creating virtual environment for Socrates"
git clone https://github.com/longph1989/Socrates.git
cd Socrates
virtualenv -p python3 socrates_venv
source socrates_venv/bin/activate
pip install numpy scipy matplotlib torch autograd antlr4-python3-runtime==4.8 sklearn pyswarms gensim python-Levenshtein
deactivate
cd ..

echo "creating virtual environment for mode"
conda create -n mode python=3.8
pip install -r ./benchmarks/mode_nn_debugging/requirements.txt
conda deactivate

echo "creating virtual environment tensorflow2"
echo "self-correcting-networks"
conda create -n scnet python
conda activate scnet
conda install tensorflow==2.3
pip install scriptify
pip install gloro
pip install tensorflow_datasets
conda deactivate

echo "creating virtual environment tensorflow1"
echo "DeepFault"
conda create -n deepfault python=3.7
conda activate deepfault
conda install tensorflow==1.13.2
conda install keras
conda deactivate

echo "creating virtual environment DeepCorrect"
conda create -n deepcorrect python=2.7
conda activate deepcorrect
conda install theano keras
pip install h5py numpy opencv convnet-keras
conda deactivate

echo "creating virtual environment diffai"
conda create -n diffAI python=3.5


echo "MinimalDNNModification"
conda create -n MinimalDNNModification python
conda activate MinimalDNNModification
conda install -c compiler
conda install cmake
conda install libgcc
pip3 install tensorflow
pip3 install numpy
git clone https://github.com/jjgold012/MinimalDNNModificationLpar2020.git
cd MinimalDNNModificationLpar2020
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd path/to/marabou/repo/folder
mkdir build 
cd build
cmake ..
cmake --build . -j 4
cd ..
cd ..
conda deactivate

echo "downloading dataset and onnx models for the benchmarks"
echo "downloading vnncomp2021 benchmarks"
git clone https://github.com/stanleybak/vnncomp2021.git

echo "downloading benchmarks and extracting zip file for Socrates"
cd benchmarks
cd AIRepair
wget "https://figshare.com/ndownloader/files/34041701?private_link=f2c4959b59cf32da4891"
mv f2c4959b59cf32da4891 benchmark.zip
unzip -d ./benchmark.zip
cd ..
cd ..
git clone https://github.com/onnx/models.git




