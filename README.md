# MFKANs: Multifidelity Kolmogorov-Arnold Networks

![MFKAN Architecture Diagram](MFKAN.png)

## Overview

**MFKANs (Multifidelity Kolmogorov-Arnold Networks)** e a low-fidelity model along with a small amount of high-fidelity data to train a model for the high-fidelity data accurately. Multifidelity KANs (MFKANs) reduce the amount of expensive high-fidelity data needed to accurately train a KAN by exploiting the correlations between the low- and high-fidelity data to give accurate and robust predictions in the absence of a large high-fidelity dataset.


## Installation

### Prerequisites
- Python 3.9.6+


MFKANs is based off jaxKAN v0.1.3 https://github.com/srigas/jaxKAN/releases/tag/v0.1.3


### Setup
```bash
# Clone the repository
git clone https://github.com/pnnl/mfkans
cd mfkans

# Install dependencies
pip install -r requirements.txt
```


## Test Cases & Examples

The repository includes 7 test cases demonstrating MFKAN capabilities

## Citation

If you use this code in your research, please cite:

```bibtex
@article{howard2025multifidelity,
  title={Multifidelity Kolmogorov--Arnold networks},
  author={Howard, Amanda A and Jacob, Bruno and Stinis, Panos},
  journal={Machine Learning: Science and Technology},
  volume={6},
  number={3},
  pages={035038},
  year={2025},
  publisher={IOP Publishing}
}
```

## Funding 

This project was completed with support from the U.S. Department of Energy, Advanced Scientific Computing Research program, under the Scalable, Efficient and Accelerated Causal Reasoning Operators, Graphs and Spikes for Earth and Embedded Systems (SEA-CROGS) project (Project No. 80278) and under the Uncertainty
Quantification for Multifidelity Operator Learning (MOLUcQ) project (Project No. 81739). The computational work was performed using PNNL Institutional Computing at Pacific Northwest National Laboratory. 

---

## DISCLAIMER

This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830

## LICENSE

Copyright Battelle Memorial Institute 2025
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
**Contact**: amanda.howard@pnnl.gov, bruno.jacob@pnnl.gov, panagiotis.stinis@pnnl.gov