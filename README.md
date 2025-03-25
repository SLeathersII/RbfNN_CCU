# RbfNN_CCU
Adapting robust model from https://github.com/AlexMeinke/certified-certain-uncertainty with an RBF which shares centers with GMMs. 
Rbf taken from: https://github.com/rssalessio/PytorchRBFLayer

Goal is to replace RBF layer with GMM, turn off the gradients, and train a classifier which then reduces confidence far from the distribution with the layer based on CCU adaptation. Implement beyond spherical variance for clusters and modify distance metrics. 
