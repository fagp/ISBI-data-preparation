set -e
apt install wget unzip
# deactivate virtualenv if already active
if command -v conda deactivate > /dev/null; then conda deactivate; fi
# python 3.8 or higher
if test -f anaconda3/envs/jcell; then
 echo "virtualenv already exists, skipping"
else
 conda create -n jcell python>=3.8
fi
# activate virtual environment
conda activate jcell
# install python dependencies or use requirements.txt
pip3 install jcell-ISBI==0.0.1a4
pip3 install -r requirements.txt
jcell-update
set +e
