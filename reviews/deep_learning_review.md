# Deep Learning Review


## Convolutional Layers

Some of the key elements of convolutional layers:
  - Invariance by translation: because you apply the same filter to the entire input, the layer can detect things (shapes, objects) no matter where they are located in the input
  - Number of parameters: convolutional layers have a very small amount of parameters compared to dense layers, so you can stack convolutional layers while still having a small amount of parameters
  - Different levels of abstraction: by stacking those layers, you get access to different level of abstraction. The first layers will detect basic features such as shapes and textures. Then the deeper you go into the stack, the more abstract the things you can detect will be. So that intermediary layers might detect parts of objects/persons (like an eye, a handle etc...) while the last layers will be able to detect whole objects. *An example of that is style transfer: you compute your loss by combining the style of a picture (ie. the first layers) with the content of a *

