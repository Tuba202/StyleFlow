# StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows (ACM TOG 2021)

## I would like to note that I had no part in the creation of this code. I am simply piecing together the works of justinjohn0306, RameenAbdal, and FlowDownTheRiver to make this program run easily on windows computers, as well as updating the installation instructions so everyone can run this awesome app!

![image](./docs/assets/teaser.png)
**Figure:** *Sequential edits using StyleFlow*

[[Paper](https://arxiv.org/pdf/2008.02401.pdf)]
[[Project Page](https://rameenabdal.github.io/StyleFlow/)]
[[Demo](https://youtu.be/LRAUJUn3EqQ)]
[[Promotional Video](https://youtu.be/Lt4Z5oOAeEY)]


## Installation

Start by making sure that you have conda(Mini or Ana, it doesn't matter), visual studio 2017, and CUDA 11 installed on your computer.
Installing Conda: https://youtu.be/tXgPY4lc6fo
Installing VS: https://youtu.be/X5zYiksQOF4
Installing CUDA: https://youtu.be/cL05xtTocmY

Now from windows search, type in Anaconda/Miniconda(whichever one you downloaded), and click on Anaconda/Miniconda powershell. Simply copy and paste in these commands:

```
conda create -y -n styleflow python==3.6.7
conda activate styleflow
conda install -y git
git clone https://github.com/Tuba202/StyleFlow-Made-Easy.git StyleFlow
cd StyleFlow/
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.2 -c pytorch
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
conda install -y tensorflow==1.14 tensorflow-gpu==1.14
pip install torchdiffeq==0.1.0 tensorflow==1.14 tensorflow-gpu==1.14 scikit-image scikit-learn requests qdarkstyle qdarkgraystyle pyqt5 opencv-python
python main.py
```
## If it doesn't work, feel free to open up an issue or email me at elliotmarks06@gmail.com. If it does, please help me out and give this project a star!!!


## License

All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**). The code is released for academic research use only.


```

## Broader Impact
*Important* : Deep learning based facial imagery like DeepFakes and GAN generated images can be gravely misused. This can spread misinformation and lead to other offences. The intent of our work is not to promote such practices but instead be used in the areas such as identification (novel views of a subject, occlusion inpainting etc. ), security (facial composites etc.), image compression (high quality video conferencing at lower bitrates etc.) and development of algorithms for detecting DeepFakes.

## Acknowledgments
This implementation builds upon the awesome work done by Karras et al. ([StyleGAN2](https://github.com/NVlabs/stylegan2)), Chen et al. ([torchdiffeq](https://github.com/rtqichen/torchdiffeq)) and Yang et al. ([PointFlow](https://arxiv.org/abs/1906.12320)). Big thanks to FlowDownTheRiver for the updated UI, justinjohn0306 for the environment file, and RameenAbdal for the rest of the code.
