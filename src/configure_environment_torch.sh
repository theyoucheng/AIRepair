#configure torch

conda create -n torch python
conda activate torch
conda install pytorch torchvision tqdm scipy scikit-learn matplotlib numpy
pip install onnx
pip install diffabs