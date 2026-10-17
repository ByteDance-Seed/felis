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
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC_BY--NC_4.0-blue"></a>
</p>

Felis (Free Energy of Ligand-protein InteractionS) is a source-available toolkit for automated and scalable protein-ligand absolute binding free energy (ABFE) calculations. It is designed for high-throughput structure-based drug discovery and supports a practical ABFE workflow without the scaffold constraints of RBFE methods.


## Getting started
### Prerequisites
* Python version >= 3.11
* CUDA Toolkit >= 12.2
* OpenMM >= 8.5, < 8.6

### Release compatibility

| Component | Version |
| --- | --- |
| FELIS | 0.2.0 |
| PerturbNonbondedPlugin | 0.1.0 |
| RelocationMCPlugin | 0.1.0 |
| OpenMM | >= 8.5, < 8.6 |
| CUDA Toolkit | >= 12.2 |

The native plugins use OpenMM CUDA implementation APIs, so the OpenMM minor
version is intentionally constrained. This release was validated with OpenMM
8.5 and CUDA Toolkit 12.2 on NVIDIA L20 GPUs.

### Dependencies

**OpenMM**

You can install OpenMM using `conda`:

```bash
conda install -c conda-forge "openmm=8.5" "cuda-version=12.2"
```

**Custom OpenMM plugins**

FELIS v2 HREMD uses `PerturbNonbondedPlugin`. Install it into the selected
OpenMM/Python environment before installing FELIS:

```bash
cd submodule/perturb-nonbonded-force
./install.sh
cd ../..
```

`RelocationMCPlugin` is required only when `rmc_enable: true`. It depends on an
already installed PerturbNonbondedPlugin and must be installed second:

```bash
cd submodule/openmm-relocationmc-plugin
./install.sh
cd ../..
```

Both plugins require CUDA. Their installers fail early if `nvcc`, OpenMMCUDA,
or the required pre-installed PerturbNonbonded libraries are unavailable.

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

After installing the dependencies and required native plugin, install FELIS:

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
The example configuration includes a documented RelocationMC section with
`rmc_enable: false`. Enable it only after installing RelocationMCPlugin; the
setting applies only to the protein/sysB leg.

## License
- FELIS-owned code and documentation are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/), except where a file or directory states otherwise.
- `submodule/openmm-relocationmc-plugin/` is licensed under [CC BY-NC 4.0](submodule/openmm-relocationmc-plugin/LICENSE).
- `submodule/perturb-nonbonded-force/` is licensed under the [GNU Lesser General Public License v3.0 or later](submodule/perturb-nonbonded-force/LICENSE). Portions adapted from OpenMM 8.5.0 `NonbondedForce` retain their embedded MIT or LGPL notices.
- `pl_bfe_dataset/` and its subdirectories retain their own license files. In particular, the dataset root is CC BY 4.0, while some included collections use Apache-2.0 or MIT terms.
- Third-party portions retain the copyright and license notices included in their source files and in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

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
