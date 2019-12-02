# Deep Learning Review


TODO: initializations

TODO: optimizers

TODO: RNNs (talk about the gates, and the different activations depending on the role)

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


# Gradient descent:

The idea of training a neural network is to find the minimum of the loss function (since that's when the network does the minimum amount of errors). To do that, you would need to take the derivative of the loss wrt each weight of the network. However, this is intractable so we can't compute the analytical solution. For this reason, we have to do a gradient descent on the loss. 

*Note that gradient descent will find the minimum of the loss only if the loss is a convex function. For a neural network, this is almost never the case (since we introduce many non-linearities through activations). Because it is non-convex, a) the gradient descent will only find a local minimum (which is good enough for deep learning), b) we lose any convergence guarantee, c) the training is sensitive to weights initialization (which is more or less a consequence of b) )*

For all variations of the gradient descent, you still do the same step:
- Draw a sample of points and targets (the size of the sample depends on the variation)
- Do a forward pass in the network for all points and generate a prediction
- Generate the loss for the sample
- Generate the gradient of the loss with regards to the weights
- Update the weights

Three different flavors:
- Batch gradient descent: the sample is the entire dataset
- Stochastic gradient descent: the sample is a single point in the dataset
- Minibatch gradient descent: the sample is a very small batch (from 32 to 256 points) 

Batch gradient descent will generate the most accuracte updates to the weights but it computationaly expensive. Stochastic gradient descent is the exact opposite. Minibatch gradient descent is a good compromise between accuracy of the update and speed. 

# Weights initialization

###### Constant value
If a constant value is chosen for initializing the weights, then all the weights for a layer will generate the same loss gradient, and they will therefore evolve symetrically (which defeats the point of having a neural net).

###### Too big
If you initialize your weights with values that are too high, the resulting gradients are very high (with possible exploding gradients), and the loss is likely to diverge.

###### Too small
On the contrary, if you initialize the weights with values that are too small, the gradients of the loss with respect to the weights will be very small (with probably some gradient vanishing), and the whole learning procedure will be slower.

###### Correct initialization: Xavier initialization
To prevent vanishing/exploding gradients, there are two rules of thumb:
- the mean of the activations should be 0
- the variance across each layer should remain the same

For the Xavier initialization, this results in having all the weights sampled from a normal distribution with mean 0 and with variance (1/(#of units in the layer)), and biases set to 0.

The choice of the variance is made so that each layer has the same variance even though there are inter-connections between those layers. With some calculus, it can be shown that the variance of the output is the product of the variances of each layer weighted by the number of units for each layer. As a result, we can show that the variance of the output is equal to the variance of the input for the Xavier initialization (i.e no exploding/vanishing gradients).

