# StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows (ACM TOG 2021)

### I would like to note that I had no part in the creation of this. I am simply piecing together the works of justinjohn0306, RameenAbdal, and FlowDownTheRiver to make this program run easily on windows computers, as well as updating the installation instructions so everyone can run this awesome app!

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

Now, from the windows search, Anaconda/Miniconda(whichever one you downloaded), and click on Anaconda/Miniconda powershell. Now simply type in these commands:

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
Remove-Item C:\Users\ellio\StyleFlow\dnnlib\tflib\custom_ops.PY
Copy-Item -Path C:\Users\ellio\Documents\StyleflowFiles\custom_ops.py -Destination C:\Users\ellio\StyleFlow\dnnlib\tflib\
python main.py
```


## License

All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**). The code is released for academic research use only.

## Citation
If you use this research/codebase/dataset, please cite our papers.
```
@article{abdal2020styleflow,
  title={Styleflow: Attribute-conditioned exploration of stylegan-generated images using conditional continuous normalizing flows},
  author={Abdal, Rameen and Zhu, Peihao and Mitra, Niloy and Wonka, Peter},
  journal={arXiv e-prints},
  pages={arXiv--2008},
  year={2020}
}
```
```
@INPROCEEDINGS{9008515,
  author={R. {Abdal} and Y. {Qin} and P. {Wonka}},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  title={Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?}, 
  year={2019},
  volume={},
  number={},
  pages={4431-4440},
  doi={10.1109/ICCV.2019.00453}}
```

## Broader Impact
*Important* : Deep learning based facial imagery like DeepFakes and GAN generated images can be gravely misused. This can spread misinformation and lead to other offences. The intent of our work is not to promote such practices but instead be used in the areas such as identification (novel views of a subject, occlusion inpainting etc. ), security (facial composites etc.), image compression (high quality video conferencing at lower bitrates etc.) and development of algorithms for detecting DeepFakes.

## Acknowledgments
This implementation builds upon the awesome work done by Karras et al. ([StyleGAN2](https://github.com/NVlabs/stylegan2)), Chen et al. ([torchdiffeq](https://github.com/rtqichen/torchdiffeq)) and Yang et al. ([PointFlow](https://arxiv.org/abs/1906.12320)). This work was supported by Adobe Research and KAUST Office of Sponsored Research (OSR). Big thanks to FlowDownTheRiver for the updated UI, justinjohn0306 for the env file, and RameenAbdal for the rest of the code.
