# GenTL 

This repository provides the code for the paper **GenTL: A General Transfer Learning Model for Building Thermal Dynamics**, available on [ArXiv](http://arxiv.org/abs/2501.13703) and soon in the ACM Digital Library. Below are instructions for using the code.

### Getting Started

To get started, clone the repository.  

``` bash
git clone https://github.com/fabianraisch/GenTL.git
```  
And change directory

``` bash
cd gentl
```
Before running the code, you first need to set up a conda environment with the dependencies needed. First, install [Anaconda](https://www.anaconda.com/download) if you haven’t already. Then, run:

```bash
conda env create -f requirements.yml
```

and activate the environment
```bash
conda activate gentl_env
```
Alternatively, you can install dependencies manually with pip by running:
```bash
pip install -r requirements.txt
```

If you want GPU support for model training, you can use  
```bash
pip uninstall torch torchvision torchaudio -y
```
and then
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```
We used CUDA version [11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) with this code.  


### Basic Usage

We provided the code to reproduce the results of our paper. *pretrain_model.ipynb* demonstrates the code for pretraining the general source model. In *heat_map.ipynb*, we demonstrate fine-tuning the general source model and the comparison with traditional single-source to single-target TL. Other functionalities are outsourced into functions that can be found in the ```src``` folder.  

### License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


### Citation
This code is based on the publication **GenTL: A General Transfer Learning Model for Building Thermal Dynamics**, available on [ArXiv](http://arxiv.org/abs/2501.13703) and soon in the ACM Digital Library.

For citation please use:  

```
@inproceedings{
    GenTL2025,
    title={GenTL: A General Transfer Learning Model for Building Thermal Dynamics},
    author={Fabian Raisch and Thomas Krug and Christoph Goebel and Benjamin Tischler},
    year= {2025}
    note= {This is the author's version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record will be published in ACM e-Energy 2025}
}
```
