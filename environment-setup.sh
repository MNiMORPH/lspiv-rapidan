
conda create --name lspiv-env
conda activate lspiv-env

# Local install of pip in the environment
conda install pip

# Stabilize video: try two options
pip install vidstab
pip install stabilo
pip install ipython # Seems I need a local ipython for the environment

# Use images from 2024_11Nov15
# https://drive.google.com/drive/folders/163PCl6BbM4UsYMQQXgibjwf3u4eVkbyJ
# Start with MAX_0102.MP4: Nice and short, and with a rotation to correct for

