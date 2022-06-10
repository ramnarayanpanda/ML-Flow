conda create --prefix ./env python=3.7 -y 
source C:/Users/Ram.Panda/Anaconda3/etc/profile.d/conda.sh 
source activate ./env 
pip install -r requirements.txt
conda env export > conda.yaml