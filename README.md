# jaxeffort
[![arXiv](https://img.shields.io/badge/arXiv-2307.14339-b31b1b.svg)](https://arxiv.org/abs/2307.14339)

Repo containing the jaxeffort emulator.

## Installation and usage

In order to install `jaxeffort`, clone this repository, enter it, and run

```bash
pip install .
```

In order to use the emulators, you have to import `jaxeffort` and load a trained emulator

```python3
import jaxeffort
import jax.numpy as np
trained_emu = jaxeffort.load_multipole_emulator("/path/to/emu/")
```
Then you are good to! You have to create an input array and retrieve your calculation result

```python3
input_array = np.array([...]) #write in the relevant numbers
result = trained_emu.get_Pl(input_array)
```

For a more detailed explanation, check the tutorial in the `notebooks` folder, which also shows a comparison with the standard `calculations.
