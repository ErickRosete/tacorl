

# TACO-RL # 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**Latent Plans for Task-Agnostic Offline Reinforcement Learning**](https://arxiv.org/pdf/2209.08959.pdf)

[Erick Rosete-Beas](https://www.erickrosete.com/), [Oier Mees](https://www.oiermees.com/), [Gabriel Kalweit](https://nr.informatik.uni-freiburg.de/people/gabriel-kalweit), [Joschka Boedecker](https://nr.informatik.uni-freiburg.de/people/joschka-boedecker), [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)


We present **TACO-RL** (**T**ask-**A**gnosti**C** **O**ffline
**R**einforcement **L**earning), a hierarchical general-purpose agent that absorbs uncurated, unlabeled, highly diverse, offline data. TACO-RL makes sense of this data by combining the strengths of imitation learning and offline RL.
We combine a low-level policy that learns latent skills via imitation and a high-level policy learned from offline RL for skill-chaining the latent behaviors.
We learn a single multi-task visuomotor policy for over 25 tasks in the real world that outperforms state-of-the-art baselines by an order of magnitude.
![](media/teaser.gif)

# :computer:  Quick Start
The package was tested using python 3.7 and Ubuntu 20.04 LTS. <br/>
```bash
# At root of project
git clone https://github.com/ErickRosete/tacorl.git
export TACORL_ROOT=$(pwd)/tacorl
conda create -n tacorl_venv python=3.7 
conda activate tacorl_venv
sh install.sh
 ```
This script install all the required dependencies for this repository.
If you want to download the dataset follow 

## Download
### Task-Agnostic Real World Robot Play Dataset
We host the multimodal 9 hours of human teleoperated [play dataset on kaggle](https://www.kaggle.com/datasets/oiermees/taco-robot).
```
cd $TACORL_ROOT/dataset
kaggle datasets download -d oiermees/taco-robot
```
### CALVIN Dataset
If you want to train on the simulated [CALVIN](https://github.com/mees/calvin) dataset, we augment the original dataset in order to balance the task distribution:
```bash
cd $TACORL_ROOT/dataset
wget http://tacorl.cs.uni-freiburg.de/dataset/taco_rl_calvin.zip && unzip taco_rl_calvin.zip && rm taco_rl_calvin.zip
```

##	:weight_lifting_man: Train TACO-RL Agent
### Simulation
To run the LMP training 
```bash
python scripts/train.py experiment=play_lmp_for_rl data_dir="dataset/calvin"
 ```

To run the TACO-RL training 

```bash
python scripts/train.py experiment=tacorl data_dir="dataset/calvin" module.play_lmp_dir="models/lmp_calvin"
 ```
### Real world
To run the LMP training 
```bash
python scripts/train.py experiment=play_lmp_real_world data_dir="dataset/real_world" 
 ```

To run the TACO-RL training 

```bash
python scripts/train.py experiment=tacorl_real_world data_dir="dataset/real_world" module.play_lmp_dir="models/lmp_real_world"
 ```

## :trophy: Evaluation
To run our experiments evaluation on the calvin environment you can use the following script
```bash
python scripts/evaluate.py evaluation=tacorl_easy module_path="PATH_MODEL_TO_EVAL" 
 ```
You can choose between the following options
- `tacorl_easy`: Single goal tasks where the goal image contains the end effector achieving the task.
- `tacorl_hard`: Single goal tasks where the goal image does not contain the end effector achieving the task.
- `tacorl_lh_easy`: Perform two tasks in a row using a single goal image.
- `tacorl_lh_seq_easy`: Perform five tasks in a row using intermediate goal images.
Analogous evaluation configurations are available for CQL, LMP and RIL.

### :student: Pre-trained Model
Download the [TACO-RL model checkpoint](http://calvin.cs.uni-freiburg.de/model_weights/tacorl_calvin.zip) trained on the static camera rgb images on CALVIN environment D.
```
$ wget http://tacorl.cs.uni-freiburg.de/model_weights/tacorl_calvin.zip
$ unzip tacorl_calvin.zip
```

## Acknowledgements

This work uses code from the following open-source projects and datasets:

#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)


## :writing_hand: Citation

If you find the dataset or code useful, please cite:

```bibtex
@inproceedings{rosete2022tacorl,
author = {Erick Rosete-Beas and Oier Mees and Gabriel Kalweit and Joschka Boedecker and Wolfram Burgard},
title = {Latent Plans for Task Agnostic Offline Reinforcement Learning},
journal = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
year = {2022}
}
```

## Contributing
If you want to contribute to the repo it is important to install
git hooks in your .git/ directory, such that your code complies with
the black code style and Flake8 style guidelines. </br>
In the root directory run: </br>
`pre-commit install`
