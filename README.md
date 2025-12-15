# navigation-test
```bash
conda create -n vln python=3.9 -y && conda activate vln
conda install -c nvidia cuda-toolkit=12.4 cuda-nvcc=12.4

conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

pip install -e .
```
安装[flash-atten-2.7.3](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl)
```bash
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```
