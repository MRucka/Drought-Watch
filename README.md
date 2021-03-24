## ConvNet drought conditions prediction using multispectral satellite images

This repo is a record of our (me along with my 2 university friends) struggles to take first place in 
[Weights & Biases "Drought Watch" community benchmark](https://wandb.ai/wandb/droughtwatch/benchmark), 
but more importantly - to **develop a robust model capable of monitoring drought conditions in Kenya!**  
<br>
I strongly encourage you to check out and contribute to Weights & Biases community benchmarks! 

## Our ideas
Since the beginnig, we wanted to do something special. <br>
First of all, we developed a **handcrafted clouds detection method** in order to discard the most cloudy and noisy images using dependencies 
between spectra (concretely, between Thermal Infrareds and Blue - we assume that such images should be marked properly and not used when assessing). 
<br>
We have also started developing a **handcrafted method for shadows removal** based on 
[this article](https://www.researchgate.net/publication/274563892_Shadow_Detection_and_Removal_from_a_Single_Image_Using_LAB_Color_Space). 
However, only **RGB shadow removal has been implemented so far and it doesn't increase model's accuracy**.
<br>
See [this notebook](./notebooks/clouds_shadows_detection) for neat visualization of methods mentioned above.
<br>
However, we hope that the biggest possible game changer will be applying ideas from [this article](https://arxiv.org/pdf/1911.07747.pdf). 
The authors state that **ConvNets are not capable of learning some useful global features**. They precalculate features and inject them in the last FC layers. <br>
A [script](../tests/haralick.py) for calculating the features has been already implemented, but calculations are presently too slow to use 
it while training on account of no vectorization (100ms vs 2s).

## What we have achieved so far
We've developed a high-accuracy model by searching for the best hyperparameters and utilizing clouds detection mentioned above.

## Environment
The model was developed and trained using TensorFlow 2 and Google Colab Tesla P100-PCIE-16GB.

## What's next
In the near future, we are going to try:
* EfficientNet;
* haralick.py vectorization;
* Data Augmentation techniques.
