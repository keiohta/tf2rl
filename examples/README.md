# Installation

## MuJoCo

If you use Anaconda, you can install MuJoCo in the following way:

```bash
$ mkdir ~/.mujoco
$ cd ~/.mujoco

# Download official package
$ wget https://www.roboti.us/download/mujoco200_linux.zip
$ unzip mujoco200_linux.zip
$ cp -r mujoco200_linux mujoco200

# Locate your license key
$ cp /path/to/mjkey.txt ./

# Copy libgcrypt11 to your Anaconda lib directory
$ wget http://archive.ubuntu.com/ubuntu/pool/main/libg/libgcrypt11/libgcrypt11_1.5.3-2ubuntu4_amd64.deb
$ dpkg-deb -xv libgcrypt11_1.5.3-2ubuntu4_amd64.deb .
$ cp ./lib/x86_64-linux-gnu/libgcrypt.so.11 ${CONDA_PREFIX}/lib
$ rm -rf ./lib ./usr
$ rm libgcrypt11_1.5.3-2ubuntu4_amd64.deb

# Install dependencies
$ conda install Cython imageio cffi lockfile patchelf flake8
$ conda install -c menpo osmesa
$ conda install -c conda-forge emacs glfw

# Install and build mujoco_py
$ CPATH=$CONDA_PREFIX/include:$CPATH pip install mujoco_py
$ CPATH=$CONDA_PREFIX/include:$CPATH python -c "import mujoco_py"
```

## Deep Mind Control Suites

Install `dm_control`.

```bash
$ cd /path/to/your/workspace
$ git clone https://github.com/deepmind/dm_control.git
$ cd dm_control
$ pip install .
```

Install `dmc2gym`.

```bash
$ cd ~/workspace/rl
$ git clone https://github.com/denisyarats/dmc2gym.git
$ cd dmc2gym
$ pip install .
```

Try `dmc2gym` sample to check if installation finishes correctly.

```bash
$ cat temp.py
import dmc2gym

env = dmc2gym.make(domain_name='point_mass', task_name='easy', seed=1)
done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
print("Finished correctly")
```

