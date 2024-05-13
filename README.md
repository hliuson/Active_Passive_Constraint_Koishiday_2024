# Koishiday's 2024: Quantifying and Optimizing Global Faithfulness in Persona-driven Role-playing

# Preparation

Before your journey, you need to prepare the following stuff:

1. Download MiniConda3 following the instructions on this [page](https://docs.anaconda.com/free/miniconda/miniconda-install/)

2. Create an OpenAI account and get an OpenAI API following the instructions on this [page](https://openai.com/index/openai-api/).

3. Create a Huggingface account and create a Huggingface Token for reading following the instructions on this [page](https://huggingface.co/settings/tokens).

4. Gain access to Gemma models following the instructions on this [page](https://huggingface.co/google/gemma-1.1-7b-it).

Then you can create an environment and install the required Python packages:

```bash
conda create -n apc python=3.8
conda activate apc
python -m pip install -r requirements.txt
```

# Quick Start

I have formalized the learning scenario for the most faithful persona-driven role-playing agent as a simple bash command. You only have to replace the ```openai_key``` and ```hf_token``` in ```bash_is_all_you_need.sh``` with your own, and then run
```bash
bash bash_is_all_you_need.sh
```

You can find the LoRA weights of the PRP agent in ```prp_models```, the intermediate datasets in ```statement```, and intermediate discriminators in ```discriminators```.
