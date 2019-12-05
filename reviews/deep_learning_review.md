# Deep Learning Review

The main point of this markdown (and the reason for the format) is to provide "oral" explanations for different aspects of deep learning. Most of the time, a schema is worth a 10-minute explanation,  but you can't draw schema on the phone. So you have to get used to explaining things with words rather than formulas and schemas.

TODO: optimizers

TODO: Overflow, underflow, and activations

TODO: RNNs (talk about the gates, and the different activations depending on the role)

# Convolutional Layers

Some of the key elements of convolutional layers:
  - Invariance by translation: because you apply the same filter to the entire input, the layer can detect things (shapes, objects) no matter where they are located in the input
  - Number of parameters: convolutional layers have a very small amount of parameters compared to dense layers, so you can stack convolutional layers while still having a small amount of parameters
  - Different levels of abstraction: by stacking those layers, you get access to different level of abstraction. The first layers will detect basic features such as shapes and textures. Then the deeper you go into the stack, the more abstract the things you can detect will be. So that intermediary layers might detect parts of objects/persons (like an eye, a handle etc...) while the last layers will be able to detect whole objects. *An example of that is style transfer: you compute your loss by combining the style of a picture (ie. the first layers) with the content of another picture (i.e the deeper layers)*
  
# RNN layers

Main idea behind the introduction of RNNs: the input might have a sequential nature (meaning that input[t] depends on some previous parts of the input). Dense layers can't handle that, and convolutional layers can also model short relationship in a fixed window. What Rnn can do is that they can model long-term relationships between parts of the input (for instance, the relationship between the current word and the previous words).

###### How it works:

An Rnn layer is basically a for loop: for each part of the input, you do a forward pass in the layer, generate an internal state (which is what conveys information between iterations) and generate an output for the current iteration. Note that the weights are shared for timesteps (you always use the same weight matrix, and you update the same matrix for all the iterations).

###### Problems:
Rnn suffer from vanishing/exploding gradients: for vanishing gradients, the long-term relationship are never actually learned because the gradient for long-term relationship (the gradients that come from deep elements in the input) are too small. This is why RNN are actually failing.

The exploding gradient issue can be fixed by manually clipping the gradient: if the gradient is above a threshold, clip it at the threshold. However, vanishing gradients cannot be fixed that way.

###### LSTM:
Its architecture solve the vanishing gradient issue by adding a cell state to the mix.

Rundown of the transformation:
- New input is first combined with previous internal state through sigmoid(mat_mul) to compute the __forget_gate__ and the __input_gate__ which define what is kept from the previous cell state and what is added from the tilde cell state
- tilde state cell is computed through combined internal state and input using tanh
- new state cell is old_state_cell\*forget_gate + tilde_state_cell\*input_gate
- new internal state is output_gate\*Cell_state where output_gate is tanh(matmul()) using previous internal state and input

As a general rule, tanh activation are used when outputing a matrix (like cell state or internal state), sigmoid when outputing a gate.

Structurally, the cell state is what helps the gradient flowing back: the only transformations it goes through are linear (no activations). It is somewhat similar to what residual connections do for convolutional nets.

GRU are a more recent alternatives to RNN: one gate less than LSTM, faster (less parameters) but usually worse in terms of performance.

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


# Optimizers

#### Challenges with optimizers

- The learning rate value is very important: if it is too high, the loss will oscillate but never converge. On the contrary, if it is too low, the convergence will be very slow
- Learning rate schedules can be used to reduce the learning rate at pre-defined timesteps or a threshold on the loss, but those values need to be chosen for each distinct dataset

#### Momentum

The idea behind momentum is to avoid having the loss oscilate while descending (think of a ravine, where you would oscilate from one side to the other, but still going deeper at every time step). Think of momentum as rolling a ball down a ravine. 

Momentum is done by adding to the current update a part of the previous update (usually around 0.9) which ensures that the new update will still keep some of the directions of the previous one. 
Momentum has two distinct effects:
- reduces the oscillation during the descent
- speeds up convergence (if two updates are in the same direction, then the addition will increase the update value)

#### Nesterov accelerated gradient

Momentum is unsatisfactory because it blindly follows a direction without anticipating where the update will leave the ball. NAG tries to correct that by changing the way we compute the gradient. What NAG does is take a big jump in the direction of the previous update, calculate the gradient at this position to compute a small correction, and add the correction to the big jump. It is better because the gradient is computed at a more precise location than for momentum.

#### AdaGrad

Another issue not covered by the two previous techniques is that some features are less frequent than others, and the parameters associated with those rare features will not be often updated. Adagrad basically ensure that the learning rate is higher for unfrequent-feature parameters than frequent ones. The bigger the updates a parameter have, the smaller the learning rate gets (there's the past squared gradients in the denominator).

A limitation with Adagrad is that the learning rates will become increasingly smaller: at some point, the updates will be infinitesimally small and the neural net will stop learning.

#### Adam

Currently one of the most used optimizer. Similarly to Adagrad, it computes learning rate wrt. each parameter, and combine past squared gradient (like RMSprop) and past gradients (like momentum) to compute updates.
