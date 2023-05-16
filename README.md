[![arXiv](http://img.shields.io/badge/arXiv-2001.09136-B31B1B.svg)](https://arxiv.org/abs/2209.01121)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Back-to-Bones
[Rediscovering the Role of Backbones in Domain Generalization](https://arxiv.org/abs/2209.01121)

<p>
    <img src="figures/scheme.png" alt= “Scheme” width="450">
</p>

*Our experimentation proves the importance of backbones in Domain Generalization. We find that novel architectures, such as transformed-based models, lead to a better representation of data, outperforming outdated backbones, such as ResNets,
and leaving marginal room for feature mapping improvement using DG algorithms.*

## Abstract
Domain Generalization (DG) studies the capability of a deep learning model to generalize to out-of-training distributions. In the last decade, literature has been massively filled with training methodologies that claim to obtain more abstract and robust data representations to tackle domain shifts. Recent research has provided a reproducible benchmark for DG, pointing out the effectiveness of naive empirical risk minimization (ERM) over existing algorithms. Nevertheless, researchers persist in using the same outdated feature extractors, and no attention has been given to the effects of different backbones yet. In this paper, we start back to the backbones proposing a comprehensive analysis of their intrinsic generalization capabilities, which so far have been ignored by the research community. We evaluate a wide variety of feature extractors, from standard residual solutions to transformer-based architectures, finding an evident linear correlation between large-scale single-domain classification accuracy and DG capability. Our extensive experimentation shows that by adopting competitive backbones in conjunction with effective data augmentation, plain ERM outperforms recent DG solutions and achieves state-of-the-art accuracy. Moreover, our additional qualitative studies reveal that novel backbones give more similar representations to same-class samples, separating different domains in the feature space. This boost in generalization capabilities leaves marginal room for DG algorithms. It suggests a new paradigm for investigating the problem, placing backbones in the spotlight and encouraging the development of consistent algorithms on top of them.

<p>
    <img src="figures/linear.png" alt= “Linear” width="600">
</p>

*DG accuracy achieved by tested backbones compared with their performance on ImageNet, with error bars. Regardless of different architectures and priors, we find a strong linear correlation between the two metrics (ρ = 0.921).*

## Description
This repository allows to train several backbones on different Domain Generalization datasets. The latest and most promising DG algorithms can be tested on the backbones to assess their efficacy.

## Installation
We suggest to use a virtual environment (conda, venv, ...)
```
git clone git@github.com:PIC4SeR/Back-to-Bones.git
pip install -r requirements.txt
```

## Usage
### Configuration
To set all the training/testing configuration parameters can be set using the arguments described the file
```
utils/utils.py
```

### Benchmark
Run the following command to benchmark backbones on different DG datasets:
```
sh benchmark.sh
```

## Results
<p>
    <img src="figures/results.png?raw=True" alt= “Results” width="1000">
</p>

Baselines comparison of different backbones for DG. We report the average accuracy over three runs and the associated standard deviation for each model. We include the results achieved by DomainBed with ResNet50 for reference. The models marked with * are pretrained on Imagenet21K instead of ImageNet1K. The rightmost column indicates the accuracy of the networks on ImageNet1K.

## Acknowledgments
This repository is intended for research scopes. If you use it for your research, please cite our paper using the following BibTeX:
```
@article{angarano2022back,
  title={Back-to-bones: Rediscovering the Role of Backbones in Domain Generalization},
  author={Angarano, Simone and Martini, Mauro and Salvetti, Francesco and Mazzia, Vittorio and Chiaberge, Marcello},
  journal={arXiv preprint arXiv:2209.01121},
  year={2022}}
```
We would like to thank the Interdepartmental Center for Service Robotics [PIC4SeR](https://pic4ser.polito.it), Politecnico di Torino.
