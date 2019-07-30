![AIcrowd-Logo](https://raw.githubusercontent.com/AIcrowd/AIcrowd/master/app/assets/images/misc/aicrowd-horizontal.png)

# Flatland Challenge Starter Kit

[![gitter-badge](https://badges.gitter.im/AIcrowd-HQ/flatland-rl.png)](https://gitter.im/AIcrowd-HQ/flatland-rl)

![](https://i.imgur.com/9cNtWjs.gif)

Instructions to make submissions to the [SBB CFF Flatland Challenge](https://www.aicrowd.com/challenges/flatland-challenge).

Participants will have to submit their code, with packaging specifications, and the evaluator will automatically build a docker image and execute their agent against an arbitrary number of pre-generated flatland environments.

### Setup

- **docker** : By following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- **aicrowd-repo2docker**

```sh
pip install aicrowd-repo2docker
```

- **Anaconda** (By following instructions [here](https://www.anaconda.com/download)) At least version `4.5.11` is required to correctly populate `environment.yml`.
- **flatland-rl** (By following instructions [here](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/readme.html))
  **IMPORTANT** : Please note that you will need `flatland-rl` version `>=0.3.9` to be able to submit which you can update in your anaconda env by :

```
pip install -U flatland-rl
```

- **Your code specific dependencies**

```sh
# If say you want to install PyTorch
conda install pytorch torchvision -c pytorch
```

### Clone repository

```
git clone git@github.com:AIcrowd/flatland-challenge-starter-kit.git
cd flatland-challenge-starter-kit
```

### Test Submission Locally

* First lets begin by downloading a small set of test envs, and put them at a location of your choice.
In this exercise, we assume that you will download the test-envs provided at : [https://www.aicrowd.com/challenges/flatland-challenge/dataset_files](https://www.aicrowd.com/challenges/flatland-challenge/dataset_files), and will untar them inside `./scratch/test-envs`, so that you have a directory structure similar to : 

```
.
└── test-envs
    ├── Test_0
    │   ├── Level_0.pkl
    │   └── Level_1.pkl
    ├── Test_1
    │   ├── Level_0.pkl
    │   └── Level_1.pkl
    ├── Test_2
    │   ├── Level_0.pkl
    │   └── Level_1.pkl
    ├── Test_3
    │   ├── Level_0.pkl
    │   └── Level_1.pkl
    ├── Test_4
    │   ├── Level_0.pkl
    │   └── Level_1.pkl
    ├── Test_5
    │   ├── Level_0.pkl
    │   └── Level_1.pkl
    ├── Test_6
    │   ├── Level_0.pkl
    │   └── Level_1.pkl
    ├── Test_7
    │   ├── Level_0.pkl
    │   └── Level_1.pkl
    ├── Test_8
    │   ├── Level_0.pkl
    │   └── Level_1.pkl
    └── Test_9
        ├── Level_0.pkl
        └── Level_1.pkl
```

* `redis-server` : **NOTE** : Please ensure that you have a redis-server running on localhost.
You can find more instructions on how to run redis [here](https://redis.io/topics/quickstart)

* **Run evaluator**

```
cd flatland-challenge-starter-kit

# In a separate tab : run local grader
flatland-evaluation --tests <path_to_your_tests_directory>

```

* **Run Agent(s)**

```
# In a separate tab :
export AICROWD_TESTS_FOLDER=<path_to_your_tests_directory>
# or on Windows :
# 
#  SET AICROWD_TESTS_FOLDER=<path_to_your_tests_directory>
python run.py
```

# How do I specify my software runtime ?

The software runtime is specified by exporting your `conda` env to the root
of your repository by doing :

```
conda env export --no-build > environment.yml
```

This `environment.yml` file will be used to recreate the `conda environment` inside the Docker container.
This repository includes an example `environment.yml`

# What should my code structure be like ?

Please follow the structure documented in the included [run.py](https://github.com/AIcrowd/flatland-challenge-starter-kit/blob/master/run.py) to adapt
your already existing code to the required structure for this round.

## Important Concepts

### Repository Structure

- `aicrowd.json`
  Each repository should have a `aicrowd.json` with the following content :

```json
{
  "challenge_id": "aicrowd_flatland_challenge_2019",
  "grader_id": "aicrowd_flatland_challenge_2019",
  "authors": ["your-aicrowd-username"],
  "description": "sample description about your awesome agent",
  "license": "MIT",
  "debug": false
}
```

This is used to map your submission to the said challenge, so please remember to use the correct `challenge_id` and `grader_id` as specified above.

If you set `debug` to `true`, then the evaluation will run on a separate set of 20 environments, and the logs from your submitted code (if it fails), will be made available to you to help you debug.

### Packaging of your software environment

You can specify your software environment by using all the [available configuration options of repo2docker](https://repo2docker.readthedocs.io/en/latest/config_files.html). (But please remember to use [aicrowd-repo2docker](https://pypi.org/project/aicrowd-repo2docker/) to have GPU support)

The recommended way is to use Anaconda configuration files using **environment.yml** files.

```sh
# The included environment.yml is generated by the command below, and you do not need to run it again
# if you did not add any custom dependencies

conda env export --no-build > environment.yml

# Note the `--no-build` flag, which is important if you want your anaconda env to be replicable across all
```

### Code Entrypoint

The evaluator will use `/home/aicrowd/run.sh` as the entrypoint, so please remember to have a `run.sh` at the root, which can instantitate any necessary environment variables, and also start executing your actual code. This repository includes a sample `run.sh` file.
If you are using a Dockerfile to specify your software environment, please remember to create a `aicrowd` user, and place the entrypoint code at `run.sh`.

## Submission

To make a submission, you will have to create a private repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).

You will have to add your SSH Keys to your GitLab account by following the instructions [here](https://docs.gitlab.com/ee/gitlab-basics/create-your-ssh-keys.html).
If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

Then you can create a submission by making a _tag push_ to your repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).
**Any tag push (where the tag name begins with "submission-") to your private repository is considered as a submission**  
Then you can add the correct git remote, and finally submit by doing :

```
cd flatland-challenge-starter-kit
# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com/:<YOUR_AICROWD_USER_NAME>/flatland-challenge-starter-kit.git
git push aicrowd master

# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change,
# then pushing a new tag will **not** trigger a new evaluation.
```

You now should be able to see the details of your submission at :
[gitlab.aicrowd.com//<YOUR_AICROWD_USER_NAME>/flatland-challenge-starter-kit/issues](gitlab.aicrowd.com//<YOUR_AICROWD_USER_NAME>/flatland-challenge-starter-kit/issues)

**NOTE**: Remember to update your username in the link above :wink:

In the link above, you should start seeing something like this take shape (the whole evaluation can take a bit of time, so please be a bit patient too :wink: ) :
![](https://i.imgur.com/4HWf1jU.png)

**Best of Luck** :tada: :tada:

# Author

Sharada Mohanty <https://twitter.com/MeMohanty>
