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
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

pip install -e .
```
