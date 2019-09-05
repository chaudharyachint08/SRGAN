.responsive 

# SuperResolution-Keras


## Problem Statement
Enhancing low resolution images by applying deep network to produce high resolutions images.


## Concept Diagram    
![Concept Diagram](images/BaseTrans.png)

    
## Reference Network Architectures
Generator & Discriminator Network of SRGAN
<img src="Architecture_images/srgan_architecture.PNG" alt="Generator & Discriminator Network of SRGAN" style="width: 100%;height: auto;" ></img>
Generator Network of E-SRGAN
![Generator Network of E-SRGAN](Architecture_images/e_srgan_architecture.PNG)
Basic Block of E-SRGAN Generator Network
![Basic Block of E-SRGAN Generator Network](Architecture_images/e_srgan_RRDB.PNG)
    

## Network Details
- k3n64s1 this means kernel 3, channels 64 and strides 1.


## Working of GAN based Super-Resolution
- We downsample HR images to get LR images
- Generator is first trained for LR to SR generation, which has to be pixelwise close to HR (high PSNR)
- Discriminator is trained to distinguish SR & HR images, loss of which is backpropagated to train Generator further
- This game theoretic process will train generator to generate SR images that are indistinguishable from HR


## Data-sets
- DIV2k and Flickr2K datasest (jointly DIV2F) for training
- Set5, Set14, BSD100, BSD300 for testing purposes

     
## Source Files
- main.py              : Contains all running procedures
- myutils.py           : Custom keras layers & function implementations
- models_collection.py : File containing all architecture definitions for fast protyping on changes


## Implementation
-[x] Data reading & LR generation, with support for heterogenous & multiformat images
-[x] Usage of Model API for model as layer, dictionary based arguments & non-comparative model training 
-[x] GAN model generator with sub-model generation, and configuring layers for model API
-[x] Disk batching for control over memory usage by limited amount of data feeding
-[x] MIX loss function with its Adaptive variant
-[x] Model parser function for easier prototyping
-[x] Plotting functions for Training progress
-[ ] Data augmentation    
-[ ] Attention Mechanism in GAN
-[ ] Network Interpolation

           
## Learning
- Residual learning helps a lot in training of deep neural networks
- Perceptual loss can be useful for domains like Image Style Transfer and Photo Realistic Style Transfer also.
- Data sollection is an import aspect, using images with high quality data, quality above quantity, as we have augmentation.


## Output
![Outputs](./images/output.PNG)


## Usage



## References
Papers
- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
https://arxiv.org/pdf/1609.04802.pdf
- ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
https://arxiv.org/abs/1809.00219
- Perceptual Losses for Real-Time Style Transfer and Super-Resolution
https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf
- Efficient Super Resolution For Large-Scale Images Using Attentional GAN
https://arxiv.org/pdf/1812.04821.pdf

Projects doing the same things
-[1] https://github.com/titu1994/Super-Resolution-using-Generative-Adversarial-Networks
-[2] https://github.com/brade31919/SRGAN-tensorflow
-[3] https://github.com/MathiasGruber/SRGAN-Keras
-[4] https://github.com/tensorlayer/srgan
 
Help on GANS
https://github.com/eriklindernoren/Keras-GAN (Various GANS implemented in Keras including SRGAN)

VGG loss help:
https://blog.sicara.com/keras-generative-adversarial-networks-image-deblurring-45e3ab6977b5
    
Improved Techniques for Training GANs:
https://arxiv.org/abs/1606.03498


## Requirements
You will need the following to run the above:
- Version of numpy 1.16.4
- Version of matplotlib 3.0.3
- Version of tensorflow 1.14.0
- Version of h5py 2.8.0
- Version of keras 2.2.5
- Version of PIL 4.3.0    
For training: Good GPU
