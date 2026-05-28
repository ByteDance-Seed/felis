<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://seed.bytedance.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/5793e67c-79bb-4a59-811a-fcc7ed510bd4">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)


# Felis
<p align="center">
  <a href="https://arxiv.org/pdf/2603.22274">
    <img src="https://img.shields.io/badge/Felis-Arxiv-red"></a>
  <a href="http://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/License-Apache-blue"></a>
</p>

Felis (Free Energy of Ligand-protein InteractionS) is an open-source toolkit for automated and scalable protein-ligand absolute binding free energy (ABFE) calculations. It is designed for high-throughput structure-based drug discovery and supports a practical ABFE workflow without the scaffold constraints of RBFE methods.


## Getting started
### Prerequisites
* Python version >= 3.11
* CUDA >= 12.6

### Dependencies

**OpenMM & OpenMMTools**

You can install OpenMM and OpenMMTools using `conda`:

```bash
conda install -c conda-forge openmm cuda-version=12.6
conda config --add channels omnia --add channels conda-forge
conda install openmmtools
conda remove jax jaxlib
```

**Gromacs**

You can easily install Gromacs using the `apt` package manager on Debian/Ubuntu-based systems:

```bash
sudo apt update
sudo apt install gromacs
```

Once the installation is complete, verify that your Gromacs version is `2022.5` or higher:

```bash
gmx --version
```

**ProLIF**

You can install ProLIF from source with the provided patch:

```bash
git clone https://github.com/chemosim-lab/ProLIF.git
cd ProLIF
git checkout v2.0.3
git apply ../submodule/prolif.patch
pip install .
```

### Installation

After resolving the dependencies above, you can install Felis and its required Python packages by running:

```bash
pip install .
```

## Quick Start Example

We provide an example ABFE calculation in the `examples/abfe/` directory. This example requires **8 GPUs** to run.

```bash
cd examples/abfe/
bash run.sh
```

The script prepares input files from `pl_bfe_dataset` and runs the full ABFE workflow.

## License
- The code portion of this project is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
- The dataset in the `pl_bfe_dataset/` directory is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Citation
If you find Felis useful for your research and applications, feel free to give us a star ⭐ or cite us using:

```bibtex
@misc{liu2026developmentlargescalebenchmarksproteinligand,
      title={Development and large-scale benchmarks of a protein-ligand absolute binding free energy toolkit}, 
      author={Yu Liu and Ailun Wang and Yu Xia and Zhi Wang and Wen Yan},
      year={2026},
      eprint={2603.22274},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2603.22274}, 
}
```

## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.

<!-- 注释：About ByteDance Seed Team可直接复制使用 -->
