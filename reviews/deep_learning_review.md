# Deep Learning Review


# Convolutional Layers

Some of the key elements of convolutional layers:
  - Invariance by translation: because you apply the same filter to the entire input, the layer can detect things (shapes, objects) no matter where they are located in the input
  - Number of parameters: convolutional layers have a very small amount of parameters compared to dense layers, so you can stack convolutional layers while still having a small amount of parameters
  - Different levels of abstraction: by stacking those layers, you get access to different level of abstraction. The first layers will detect basic features such as shapes and textures. Then the deeper you go into the stack, the more abstract the things you can detect will be. So that intermediary layers might detect parts of objects/persons (like an eye, a handle etc...) while the last layers will be able to detect whole objects. *An example of that is style transfer: you compute your loss by combining the style of a picture (ie. the first layers) with the content of another picture (i.e the deeper layers)*
  
  
# Dropout

Regularization layer to prevent overfitting.It works by randomly dropping activations from the previous layer during training (the proportion of activations dropped is a parameter of the layer). During inference, the dropout layers are ignored.

*Intuition: forcing the network to learn redundant representations of the input to make sure that it only learns the more important ones. More specifically, because at every batch certain weights are dropped, the network will learn to extract the same features with different nodes (i.e different nodes will encode the same representation). As a result, the network will only learn the most important representations (those that give good performance on the training set) and we avoid overfitting (which could correspond to layers that have learned "too much" from the training set and cannot generalize well anymore)*

# Batch Normalization

Batch normalizations are layers that will normalize each batch: it removes the batch mean and divide by the batch standard deviation for each activations of the previous layer (for each batch). 

A batch normalization layer has four different kind of parameters:
- moving average (non-trainable)
- moving standard deviation (non-trainable)
- &alpha; weights (trainable)
- &beta; weights (trainable)

&alpha; and &beta; are weights learned during training for rescaling and shifting the normalized activations using the following formula:

   y<sub>i</sub> = &alpha;x<sub>i</sub> + &beta;
                
The moving average and the moving standard deviation are used during training and inference for normalizing the activations.
Note that during inference, the moving average and moving standard deviation are fixed.

Batch normalization layers have two distinct effects:
- it improves the convergence speed: during training, the distribution of the features will vary for each batch, and the network is constantly going to try to adapt to those new distributions (this is called the __covariate internal shift__)
- it acts as regularization: by normalizing, we are introducing some noise in the data



