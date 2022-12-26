# Neural Network

A neural network from scratch.

# Formulas

## Sigmoid Function

$$ f(z) = {1 \over (1 + e^{-z})} \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; f\prime(z) = f(z) \cdot (1 - f(z)) $$

## Binary Cross-Entropy Cost Function

$$ L_{CE} = -\sum_{i=1}^n t_i log(p_i) $$

for $n$ classes, where $t_i$ is the true label and $p_i$ is the Softmax probability for the $i^{th}$ class.

# Resources

I referenced the following resources:

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Weight Initialization Techniques for Deep Neural Networks](https://www.geeksforgeeks.org/weight-initialization-techniques-for-deep-neural-networks/)
