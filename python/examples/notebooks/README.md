<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/docs/assets/Bee_logo_white.svg">
    <source media="(prefers-color-scheme: light)" srcset="/docs/assets/Bee_logo_black.svg">
    <img alt="Bee Framework logo" height="90">
  </picture>
</p>

<h1 align="center">BeeAI Framework Notebooks</h1>

This series of notebooks will guide you through the features of the BeeAI framework. You'll learn how to create, manage, and optimize AI agents with the help of practical examples and explanations.

## ✅ Pre-requisites

These notebooks use Ollama and the IBM Granite 3.1 8B LLM.

Ollama is a free, open-source tool that lets you run large language models (LLMs) locally on your computer. 🚀

You will need to have Ollama installed with the required model installed.

⚙️ Install Ollama

Follow the [Official Ollama Instructions](https://ollama.com/download/)

⚙️ Install the IBM Granite 3.1 model using

```shell
ollama pull granite3.1-dense:8b
```

⚡ If you'd prefer to use watsonx to run Granite 3.1 instead of Ollama (e.g., if your machine lacks sufficient resources), refer to the [watsonx.ipynb](watsonx.ipynb) notebook, which demonstrates how to use watsonx. You can then substitute the watsonx ChatModel when working through the main series of notebooks. ⚡

## 🛠 Setup Instructions

⚙️ First clone the beeai-framework repo.

If you frequently work with git and have ssh keys configured:

```shell
git clone git@github.com:i-am-bee/beeai-framework.git
```

Otherwise you can use https if you don't want to set up SSH keys or are in an environment where SSH connections are restricted:

```shell
git clone https://github.com/i-am-bee/beeai-framework.git
```

🧭 Next you should navigate to the python subfolder:

```shell
cd beeai-framework/python
```

🐍 You will need Python 3.11 or above. This can be checked with:

```shell
python --version
```

> [!NOTE] 
> See the [Python documentation](https://www.python.org/) for further information on installing or upgrading.

❗It is recommended to run the notebooks within a virtual environment. You can create one using the following commands:

```shell
python -m venv .venv
```

Ensure the Python environment is activated in each new terminal window:

```shell
source .venv/bin/activate
```

📦 You are now ready to install dependencies:

```shell
pip install -r examples/notebooks/requirements.txt
```

✨ And finally you can spin up the first notebook ✨

```shell
jupyter-lab examples/notebooks/basics.ipynb --port 9999
```

> [!NOTE]
> If you're not familiar with how to get started with Jupyter Notebooks you can read the [documentation](https://docs.jupyter.org). 

## Notebooks

There are three notebooks available. It is recommended to start with [basics.ipynb](basics.ipynb), followed by [workflows.ipynb](workflows.ipynb), and finally [agents.ipynb](agents.ipynb). You can navigate through the notebooks using the embedded links.

The [basics.ipynb](basics.ipynb) notebook introduces the core components of the BeeAI framework, including PromptTemplates, Messages, and Memory. It also covers setting up a ChatModel and generating output.

- If you plan to use watsonx, be sure you have reviewed the supplementary [watsonx.ipynb](watsonx.ipynb) notebook before proceeding.

The [workflows.ipynb](workflows.ipynb) notebook explains how to use BeeAI Workflows to create AI agents of varying complexity. It builds upon the concepts introduced in [basics.ipynb](basics.ipynb).

Finally, the [agents.ipynb](agents.ipynb) notebook covers setting up and using the BeeAI ReActAgent. This pre-configured ReAct agent can be customized with tools and instructions to solve various problems.

