# Mask R-CNN + DeepFashoin2
- based on tensorflow and keras
- use DeepFashion2 dataset to train Mask R-CNN
- origin code from https://github.com/Manishsinghrajput98/Deepfashion2_Training

## how to use
1. download DeepFashion2 dateset in [here](https://github.com/switchablenorms/DeepFashion2)
2. git clone this repo to local (git clone https://github.com/Xlgd/Mask_R-CNN_DeepFashion2)
3. install anaconda and create a new environment (conda create -n env_name python=3.6)
4. activate the environment (conda activate env_name)
5. install packages with requirement.txt (while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirement.txt)
6. convert DeepFashin2 json to coco format json (use tools/deepfashion2coco.py)
7. add image path and coco format json path in main.py (line 44 ~ 47)
8. begin training (python main.py)