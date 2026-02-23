# AgaeSMO

<img width="4251" height="3751" alt="architecture - 副本 (2) - 副本_画板 1" src="https://github.com/user-attachments/assets/f33d1196-8ce7-4601-b2e5-a6809ab1ccc5" />

H5ad file of result deposited in [https://drive.google.com/drive/folders/1UgXCfGyDs4GJzuc5kBD-KruUdnWwbO7U?usp=sharing](https://drive.google.com/drive/folders/1UgXCfGyDs4GJzuc5kBD-KruUdnWwbO7U?usp=sharing)

data of demo deposited in https://drive.google.com/drive/folders/1eK5zSKmSV2eaQ9jhIEcKPXwuiWaGfldH?usp=sharing

## installation

git clone https://github.com/LiuQuin/AgaeSMO.git

cd AgaeSMO



conda create -n AgaeSMO_env python=3.9 r-base=4.2.3

conda activate AgaeSMO_env

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

conda install scanpy pandas matplotlib seaborn numpy=1.26.4 ipywidgets python-louvain scikit-misc r-mclust rpy2==3.5.9

conda install anaconda::ipykernel

python -m ipykernel install --user --name AgaeSMO_env --display-name "AgaeSMO_env"

python setup.py install

## Tutorials
see Tutorials_ST_HE.ipynb && Tutorials_ST_SM.ipynb
