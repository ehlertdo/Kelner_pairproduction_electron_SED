This is a straightforward implementation of the [Kelner & Aharonian (2008)](https://arxiv.org/abs/0803.0688) analytical parametrisation of the electron spectrum for Bethe-Heitler pair production (see Eq. 62 in their paper) for arbitrary tabulated photon fields. An example photon field (CMB) and output spectrum are provided for illustration purposes. The output was checked against Fig. 10 in the paper.

The output file format is intended to be compatible with [CRPropa3-data](https://github.com/CRPropa/CRPropa3-data).

There are two branches:

(a) main: The integration loops are implemented in kelner.c and loaded into the python script via ctypes. Before running the script (`python secondary_electron_spectrum.py`), the C code must be compiled for the local machine (`gcc -fPIC -O -shared -o kelner.so kelner.c`). Execution of the script takes a few seconds for a single target proton energy.

(b) numba: Full-python implementation. Numba is used to run the script in a reasonable amount of time. Factor ~10 slower than the code with C & ctypes.
