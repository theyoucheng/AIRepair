#configure torch

echo "firstly installing anaconda"
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
