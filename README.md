# ImageClassifier
## Specifications
The project includes two files ```train.py``` and ```predict.py```. The first file, ```train.py```, will train a new network on a dataset and save the model as a checkpoint. The second file, ```predict.py```, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need.

### 1. Train
Train a new network on a data set with train.py

**Basic usage**: ```python train.py data_directory``` <br>
Prints out training loss, validation loss, and validation accuracy as the network trains <br>
**Options**: <br>
Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory``` <br>
Choose architecture: ```python train.py data_dir --arch "vgg13"``` <br>
Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_units 1024 --epochs 15``` <br>
Use GPU for training: ```python train.py data_dir --gpu``` <br>

### 2. Predict
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

**Basic usage**: ```python predict.py /path/to/image checkpoint``` <br>
**Options**: <br> 
Return top KK most likely classes: ```python predict.py input checkpoint --top_k 3``` <br>
Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json``` <br>
Use GPU for inference: ```python predict.py input checkpoint --gpu``` <br>
