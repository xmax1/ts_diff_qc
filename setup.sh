conda env remove -n m
rm -rf /opt/conda/envs/m
echo "removed /opt/conda/envs/m"
conda clean --all -y
conda create -n m python=3.10.10 -y
source ~/.bashrc
echo "source bash"
conda activate m
conda install -c conda-forge ipython ipykernel -y
ipython kernel install --user --name=m
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
python -m pip install torch torchvision torchaudio
conda install -c conda-forge -y "xarray[complete]"
conda install -c conda-forge -y dask 
conda install -c conda-forge -y matplotlib 
conda install -c conda-forge -y geoviews 
conda install -c conda-forge -y cartopy
conda install -c conda-forge -y netCDF4
conda install -c -y bokeh bokeh 
conda install -c conda-forge pygwalker