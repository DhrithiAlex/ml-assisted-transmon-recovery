# ML-Assisted Transmon Hamiltonian Recovery

**Recover $E_J$, $E_C$, and coupling $g$ from noisy synthetic spectroscopy using Gaussian Processes and Neural Networks.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Physics-accurate transmon Hamiltonian in charge basis
- Realistic noisy Lorentzian spectroscopy simulation + peak fitting
- Two models: **Gaussian Process** (with uncertainty) + **MLP**
- Noise robustness analysis
- Clean, modular, extensible design

## Installation

```bash
git clone https://github.com/DhrithiAlex/ml-assisted-transmon-recovery.git
cd ml-assisted-transmon-recovery
pip install -e .
```

## Quick Start

```bash
# Run full pipeline
transmon-recovery

# or
python scripts/run_pipeline.py
```

## Adding Case Studies

Place new Jupyter notebooks in the `notebooks/` folder.

## License
MIT © Dhrithi Alex