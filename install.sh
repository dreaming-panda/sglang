pip install setuptools==78.1.1 pip==25.1
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install nvidia-nvshmem-cu12
pip install flashinfer-python==0.2.7.post1 --no-deps --no-build-isolation
pip install -e "python[all]"
pip install pytest
pip install uvloop==0.21.0 uvicorn==0.35.0