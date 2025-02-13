# Code Autocompletion

## Installation

Download the repo on your computer

```bash
git clone https://github.com/NicoPolazzi/autocompletion.git
cd autocompletion
```

Build the virtual environment

```bash
python -m venv autocompletion_env
```

Activate the virtual environment (assuming UNIX system):

```bash
source autocompletion_env/bin/activate
```

Install the requirements

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Before running the program, you can change the values in `config.yaml` for the dataset, model and training parameters.


### Training and validate the model

To train and validate the model simply run:

```bash
python autocompletion.py train
```

### Using the model for code completion

To perform inference for code completion for a specific code snippet:

```bash
python autocompletion.py inference --snippet "def my_function("
```
