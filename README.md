# GAN-study

Hi, this repository was created as a student of the master's degree in (computer engineering? i'm not sure that is the correct translation :P), and has the propose to help me understend the topic "generative adversial network". Also the english isn´t my strong point, so i hope i can improve as well.

## GAN models

Here i present several GAN models and basic classifiers (just to compare results) in format of notebook implemented with tensorflow using the layers API

#### Current implementation's
* GAN (original 2014)
* Conditional GAN
* NN for MNIST Classification (simple nn, no dropout or other fancies tecniques, just to have baseline score)
* Auxiliar classifier GAN

#### Current/future work
* CNN for MNIST Classification (just to have baseline score)
* Deep Convolution GAN
* Semi Supervised GAN (Improved Techniques for Training GANs)
* Information Retrieval GAN

### Material used for current study
* GAN - https://arxiv.org/abs/1406.2661
* Conditional GAN - https://arxiv.org/abs/1411.1784
* Auxiliar classifier GAN - https://arxiv.org/abs/1610.09585
* Deep Convolution GAN - https://arxiv.org/abs/1511.06434
* SSGAN - https://arxiv.org/abs/1606.03498
* Book - Hands-On Machine Learning with Scikit-Learn and TensorFlow (first steps)
* Book code - https://github.com/ageron/handson-ml
* Models code examples - https://github.com/wiseodd/generative-models

### Material for future study

* EBGAN - https://arxiv.org/abs/1609.03126
* BEGAN - https://blog.heuritech.com/2017/04/11/began-state-of-the-art-generation-of-faces-with-generative-adversarial-networks/
* BEGAN - (paper)
* IRGAN - (paper)

### Requirements
* jupyter notebook
* tensorflow (1.4, 1.5 (soon)) (All the code was run on GPU version (but CPU should work to))
* numpy
* matplotlib
