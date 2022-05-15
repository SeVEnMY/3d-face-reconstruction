# 3d-face-reconstruction

### 3D Face Reconstruction with 3DMM Face Model from RGB image
This is a project reconstructing 3D face mesh from related RGB image, with the help of Basel Face Model(BFM) and soft renderer(differantiable renderer). We present a pipeline that reconstructs a human face 3D model from a single RGB image. The pipeline includes face detection, landmark detection, regression of 3DMM model parameters, and soft rendering. 

#### Test
1. Make sure you have python3 and related pip3 installed.
2. Make sure you have Anaconda3 installed. 
3. Run the following code to setup the environment:
```
conda env create -f environment.yml
source activate py3dface
```
4. Please download our pretrained model from [Google Drive](https://drive.google.com/file/d/1NfyXzh_CV-BWlZfOK7K68YLoNSu4lEgz/view?usp=sharing).
5. Please download the pretrained resnet50 model from [Google Drive](https://drive.google.com/file/d/1B3U2bdZlRh7BldGoiemxUAXs7BKKwxDJ/view?usp=sharing). and put it under folder `./checkpoints/init_model`
6. Please download the BFM from [Google Drive](https://drive.google.com/file/d/1XAGc2VcidxRGIaP0OAh3S54YIVakzroe/view?usp=sharing). and put it as './BFM'
7. Please install nvdiffrast from [Github]https://github.com/NVlabs/nvdiffrast/tree/a4e7a4db7e09695b4efc7641cc6b044ef706f953. and put it as './nvdiffrast'
8. Put the model to the folder `./checkpoints/twoloss/...`
9. Put images you want to test with into the folder `./examples`
10. Run `preprocessing.ipynb` with all images you want to test with, and put all generated `.txt` files (with the same name but different postfix as images) into the folder `./examples/detections`.
11. Run:
```
python test.py
```
9. Results will be stored in ./checkpoints/twoloss/examples
