# Speech_Workshop_ECE
Speech Processing Workshop for ECE Dept

## Installation

The codes in this Repository are run using Python(version 3.x)<br/>
The libraries used for the installation are given in the requirements.txt file

### Linux Users
For Linux based Operating systems you can use the makefile provided  
Open up a terminal and run  
```
pip3 install -r requirements.txt
```
This will install all libraries into your active python environment

### Windows Users
For Windows users, we recommend the use of Anaconda.<br/>
Link for Instructions to download Anaconda : https://docs.anaconda.com/anaconda/install/windows/

**Verify the Anaconda installation:**<br/>
1. Open Anaconda Prompt and type conda. The command should be executable and should not throw any error.
2. Now type python. This should open an interpreter in the command prompt.
3. If both commands can be exected without any error, then the installation was successful

**Install missing packages:**<br/>
Anaconda comes with many packages installed. Along with the default packages, we additionally need 2 more packages<br/>
Run:
1. pip install librosa (To install Librosa)
2. conda install -c anaconda scikit-learn (To install Scikit Learn)

### Verify packages
Run this snippet of code given below

```python
import numpy as np
import matplotlib.pyplot as plt
import librosa
import sklearn
import scipy
import pickle
```

A successful run would yield no errors and would import all the libraries correctly
