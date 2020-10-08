# fashion_mnist_recognizer
First lab of Computer Vision in Applications course. Designing a deep network (CNN) to classify Fashion-Mnist dataset [1].

## Install packages
```pip install -r requirements.txt```

## Data collection:
- Fashionmnist dataset [1] includes 60000 training images (28x28) and 1000 test images (28x28). I splitted training set to 5000 images for training and 1000 images for validation.
- Normalize data to range (0, 1) by deviding each image by 250.
- Visualize data
- Augment data (this step was not accomplished yet): 2 augmentation method were used are croping and horizontal flip.

## Step-by-step procedures to train CNN model using tensorflow:
- Init computation graph. 
- Create placeholders for training set X and Y.
- Init parameters W1 and W2 for model.
- Forward propagation.
- Compute cost
- Create optimizer to minimize cost and tensorflow will perfome back prop for us
- Init global variables for model
- Start session and execute above operators per epoch

## Model configuration
Best models are picked up based on model performance on validation set.

![Network architecture](model.png)

Model includes 2 CONV + 2 FC:
- CONV: filter size (3x3 hoặc 5x5), padding 'SAME' và activation function Relu
- Pool layer: stride = 2
- FC1: 1024-d vector
- FC2: 10-d vector

### Config 1:
Mỗi CONV block bao gồm: Conv -> Relu và Dropout được dùng như regularization trước FC2.
Each CONV block includes: 1 Conv -> 1 Relu and 1 following Dropout layer (for regularization) except for FC2.
### Config 2:
Each CONV block includes: Conv -> Relu -> BatchNorm and each Pool layer and FC1 associate with 1 Dropout layer afterwards.

Some hyper-params to fine-tune:
- kernel size: f1, f2 (3 hoặc 5)
- learning_rate: 1e-2, 1e-3, 1e-4.
- weight_decay: regularize model to avoid overfitting (default: 5e-4)
- num_epoches: number of epochs to train model (Ví dụ: 50, 80, 100)
- batch_size: 32 or 64
- keep_prob: used for dropout (0.5, 0.6) to reduce the dependence of model on some certain nodes and also to have regularization to prevent overfitting. 

## Fit model:
```python
_, _, params = fit_model(X_train, Y_train, X_val, Y_val, \
                         no_epochs=3, learning_rate = 1e-4, keep_prob_input=0.5, weight_decay=5e-4, \
                         minibatch_size=50, plot_learning_curve=True, model_name=dtime, model_number=1)
```
Where, model_number (0 or 1): indicates model configuration to use

## Evaluate:
```python
evaluate(model_meta_file=path+'/'+dtime+'-1000.meta', X_test, Y_test)
```
Arguments:
- model_meta_file: pretrained model file
- X_test: test set
- Y_test: lables of test set
  
## Run commandline
```
python 1612174.py -train "Training set files (.csv)" -test "Test set files (.csv)"
```

**Note: I provided `.ipynp` file for further clarfication and demonstration purpose.**

## Experiments:
|Model|Config|Hyperparams|Train acc|Val acc|Test acc|
|---|---|---|---|---|---|
|1|1|f1=f2=5, no_epochs=80, learning_rate=1e-4, keep_prob_input=0.5, weight_decay=5e-4, minibatch_size=32, weight_decay=5e-4|0.9998|0.9266|0.7362|
|2|2|f1=f2=5, no_epochs=80, learning_rate=1e-4, keep_prob_input=0.5, weight_decay=5e-4, minibatch_size=50, weight_decay=5e-4|0.906|0.8842|0.8406|

## Reference:
[1] Fashion-MNIST dataset: https://www.kaggle.com/zalando-research/fashionmnist