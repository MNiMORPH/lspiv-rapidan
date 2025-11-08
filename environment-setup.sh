
conda create --name lspiv-env
conda activate lspiv-env

# Local install of pip in the environment
conda install pip
# Local Jupyter and ipython
conda install jupyter
conda install ipython # Seems I need a local ipython for the environment
#conda install ipykernel
python -m ipykernel install --user --name=lspiv-env --display-name="Python (lspiv-env)"

# Stabilize video: try two options
pip install vidstab
pip install stabilo

# Use images from 2024_11Nov15
# https://drive.google.com/drive/folders/163PCl6BbM4UsYMQQXgibjwf3u4eVkbyJ
# Start with MAX_0102.MP4: Nice and short, and with a rotation to correct for

# Downgrade the Python environment for PyORC
conda install python=3.12

# RIVeR: GUI-based PIV system.
# Some by-hand cropping; maybe good for a one-off
pip install river-cli
#PyORC
#pip install pyopenrivercam
# Now installed from GitHub clone

# Extra for pyopenrivercam
conda install cartopy

# Numba within pyOpenRiverCam requires NumPy v2.2 or earlier
# In fact, really needs 1
conda install numpy=1

