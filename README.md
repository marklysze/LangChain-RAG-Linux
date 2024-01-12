# RAG with LangChain - Nvidia CUDA + Linux + Word documents + Local LLM (Ollama)

These notebooks demonstrate the use of LangChain for Retrieval Augmented Generation using Linux and Nvidia's CUDA.

Note: Using [LangChain v0.1](https://blog.langchain.dev/langchain-v0-1-0/).

Environment:
- Linux (I'm running Ubuntu 22.04)
- Conda environment (I'm using Miniconda)
- CUDA (environment is setup for 12.2)
- Visual Studio Code (to run the Jupyter Notebooks)
- Nvidia RTX 3090
- 64GB RAM (Can be run with less)
- LLMs - Mistral 7B, Llama 2 13B Chat, Orca 2 13B, Yi 34B, Mixtral 8x7B, Neural 7B, Phi-2, SOLAR 10.7B - Quantized versions

Your Data:
- Add Word documents to the "Data" folder for the RAG to use

Package versions:
- See the "conda_package_versions.txt" for the full list of versions in the conda environment (generated using "conda list").

Local LLMs:
- Ollama is run locally and you use the "ollama pull" command to pull down the models you want. For example, to pull down Mixtral 8x7B (4-bit quantized):
```
ollama pull mixtral:8x7b-instruct-v0.1-q4_K_M
```
- See the Ollama [models page](https://ollama.ai/library) for the list of models. Within each model, use the "Tags" tab to see the different versions available ([example](https://ollama.ai/library/mixtral/tags)).

Note that [nvtop](https://github.com/Syllo/nvtop) is a useful tool to monitor realtime utilisation of your GPU. Helpful to make sure the models fit within GPU memory (and don't go into your RAM and use your CPU as well).

# Installation

1. Open a terminal
2. Navigate to the directory where you want to clone the repository
3. Clone the repository
```
git clone https://github.com/marklysze/LangChain-RAG-Linux-CUDA
```
4. Navigate to the repository directory
```
cd LangChain-RAG-Linux-CUDA
```
5. Create the Conda environment
```
conda env create -f environment.yml
```
6. Activate the Conda environment
```
conda activate LangChainRAGLinux
```
5. Run Visual Studio Code
```
code .
```
6. Choose a a Jupyter Notebook file to open
7. On the top-right you may need to choose the newly created kernel. In the top-right if it says "Select Kernel", click it and choose your Python environment... and then "LangChainRAGLinux".
8. [Install Ollama](https://python.langchain.com/docs/integrations/llms/ollama) ([repository with instructions](https://github.com/jmorganca/ollama)) if you haven't already and pull the models you want to use. See above for sample command to pull models.
9. Run the Jupyter Notebook

### Output examples

Question asked of the model based on the story:
> Summarise the story for me

---
**Mistral 7B:**
> 

---
**Llama 2:**
> 

---
**Orca 2:**
> 

---
**Yi 34B:**
> 

---
**Mixtral 8X7B:**
> 

---
**Phi-2: [Quantized]**
>

---
**Phi-2: [FP16]**
>

---
**Neural Chat 7B:**
> 

---
**SOLAR 10.7B Instruct:**
> 