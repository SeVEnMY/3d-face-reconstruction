# 3d-face-reconstruction

### 3D Face Reconstruction with 3DMM Face Model from RGB image
This is a project reconstructing 3D face mesh from related RGB image, with the help of Basel Face Model(BFM) and soft renderer(differantiable renderer). 

#### Test
1. Make sure you have python3 and related pip3 installed.
2. Make sure you have Anaconda3 installed. 
3. Run the following code to setup the environment:
```
conda env create -f environment.yml
source activate py3dface
```
4. Please download our pretrained model from [Google Drive](https://drive.google.com/file/d/1NfyXzh_CV-BWlZfOK7K68YLoNSu4lEgz/view?usp=sharing).
5. Put the model to the folder `./checkpoints/twoloss/...`
6. Put images you want to test with into the folder `./examples`
7. Run `preprocessing.ipynb` with all images you want to test with, and put all generated `.txt` (with the same name but different postfix as images) into the folder `./examples/detections`.
8. Run:
```
python test.py
```
9. Results will be stored in ./checkpoints/twoloss/examples
