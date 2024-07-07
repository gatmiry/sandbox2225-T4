wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

## the next line is in case the global path to conda was not automatically set
export PATH=~/miniconda3/bin:$PATH

conda install python3

## creating virtual env
conda ceate --name env3
conda activate env3
conda deactivate

## update conda
conda update conda

## check conda
which conda

## installing other dependencies
python -m pip install jupyter
pip install accelerate -U
conda install conda-forge::transformers
conda install datasets
