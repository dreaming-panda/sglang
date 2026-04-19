pip install setuptools==78.1.1 pip==25.1

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install flashinfer-python
pip install PyYAML==6.0.3
pip install -e "python[all]"
pip install pytest
pip install uvloop==0.21.0 uvicorn==0.35.0
