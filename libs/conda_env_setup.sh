wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

## the next line is in case the global path to conda was not automatically set
export PATH=~/miniconda3/bin:$PATH

conda install python3
conda ceate --name myenv2

## update conda
conda update conda
