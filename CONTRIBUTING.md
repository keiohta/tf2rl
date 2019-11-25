## Contributing to TF2RL

Any contributions would be very welcome!

Here are contribution examples:

- Create an issue about a new feature, algorithm, bug, installation trouble, etc.
- Send pull requests about fixing bug, adding new features, adding new algorithms, etc.
- Thumbing up an outstanding issue :+1:
- Star this repository (cheers up the author :muscle:)


If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

## Developing TF2RL

To develop TF2RL on your machine, here are some tips:

1. Clone a copy of TF2RL from source:

```bash
git clone https://github.com/keiohta/tf2rl.git
cd tf2rl
```

2. Install TF2RL in develop mode, with support for running examples:

```bash
pip install -e .[examples]
```

3. TF2RL is built on Google's TensorFlow and requires either `tensorflow` or `tensorflow-gpu`.
   To include the TensorFlow with the installation of TF2RL, add the flag `tf` for the CPU version, or `tf_gpu` for the GPU version:

```bash
# Install TF2RL with TensorFlow CPU version
$ pip install -e .[examples,tf-gpu]
```

## Codestyle

We follow the [PEP8 codestyle](https://www.python.org/dev/peps/pep-0008/), and suggest the order of the imports as:

1. built-in
2. packages
3. current module

with one space between each,  that gives for instance:
```python
import os

import numpy as np

from tf2rl.algos.ddpg import DDPG
```

Also, please documentation each function/method using the following template:

```python
def my_function(arg1, arg2):
    """
    Short description of the function.

    :param arg1: (arg1 type) describe what is arg1
    :param arg2: (arg2 type) describe what is arg2
    :return: (return type) describe what is returned
    """
    ...
    return my_variable
```

## Pull Request (PR)

Before proposing a PR, please open an issue, where the feature will be discussed. This prevents from duplicated PR to be proposed and also ease the code review process.

Each PR need to be reviewed and accepted by at least one of the maintainers (Currently only @keiohta).
A PR must pass the Continuous Integration tests (travis) to be merged with the master branch.

## Test

All new features must add tests in the `tests/` folder ensuring that everything works fine.
Also, when a bug fix is proposed, tests should be added to avoid regression.

To run tests:

```bash
$ python -m unittest discover
```


Credits: this contributing guide is based on the [PyTorch](https://github.com/pytorch/pytorch/) one.

