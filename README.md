# RAG with LangChain - Nvidia CUDA + Linux + Word documents + Local LLM (Ollama)

These notebooks demonstrate the use of LangChain for Retrieval Augmented Generation using Linux and Nvidia's CUDA.

Note: Using [LangChain v0.1.1](https://blog.langchain.dev/langchain-v0-1-0/).

Environment:
- Linux (I'm running Ubuntu 22.04)
- Conda environment (I'm using Miniconda)
- CUDA (environment is setup for 12.2)
- Visual Studio Code (to run the Jupyter Notebooks)
- Nvidia RTX 3090
- 64GB RAM (Can be run with less)
- LLMs - Mistral 7B, Llama 2 13B Chat, Orca 2 13B, Yi 34B (Work in progress), Mixtral 8x7B, Neural 7B, Phi-2, SOLAR 10.7B - Quantized versions

Your Data:
- Add Word documents to the "Data" folder for the RAG to use

Package versions:
- See the "environment.yml" for the full list of versions in the conda environment (generated using "conda list").

Local LLMs:
- Ollama is run locally and you use the "ollama pull" command to pull down the models you want. For example, to pull down Mixtral 8x7B (4-bit quantized):
```
ollama pull mixtral:8x7b-instruct-v0.1-q4_K_M
```
- See the Ollama [models page](https://ollama.ai/library) for the list of models. Within each model, use the "Tags" tab to see the different versions available ([example](https://ollama.ai/library/mixtral/tags)).

Note that [nvtop](https://github.com/Syllo/nvtop) is a useful tool to monitor realtime utilisation of your GPU. Helpful to make sure the models fit within GPU memory (and don't go into your RAM and use your CPU as well).

# Notebooks

### 01-LangChain-RAG
Get started with LangChain and Ollama, being able to use various local LLMs and Word Documents as sources for Retrieval Augmented Generation (RAG). Have it answer a few questions and see what they give you.

### 02-LangChain-RAG LangSmith
To help with being able to see what is happening under the hood, sign up for a LangSmith Access Code and use this notebook to see how it is setup. Same functionality as the previous notebook.

### 03-LangChain-RAG Chunk Rerank
Get started with breaking up the document yourself into better chunks and then using Cohere's reranking (free non-commercial API key available) to prioritise the chunks for your questions. I found the self-chunking and reranking improved the LLM responses significantly.

### 04-LangChain-RAG Chunk Rerank Max Context
Continuing on from #03, we now want to maximise the amount of context given to the LLM. Previously this was a set number of chunks, now we keep track of the number of tokens per chunk and give the LLM the maximum number of chunks we can fit into a given token limit (which we set). 

We add callbacks (including a hack to workaround a bug) to count tokens and to manually choose the right amount of chunks to give the LLM after retrieval.

This maximises the amount of context given to the LLM while keeping within a set context length so we don't exceed the LLM's context window.

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

7. [Install Ollama](https://python.langchain.com/docs/integrations/llms/ollama) ([repository with instructions](https://github.com/jmorganca/ollama)) if you haven't already and pull the models you want to use. See above for sample command to pull models.

8. Run Visual Studio Code
```
code .
```

9. Choose a a Jupyter Notebook file to open

10. On the top-right you may need to choose the newly created kernel. In the top-right if it says "Select Kernel", click it and choose your Python environment... and then "LangChainRAGLinux".

11. Run the Jupyter Notebook

# Output examples

Question asked of the model based on the story:
> Summarise the story for me

---
**Llama 2:**
```
Sure! Here is a summary of the story:
 
Thundertooth, a time-traveling dinosaur, founded a toy factory in a futuristic city where he and his family produced magical widgets that brought joy to children. When a meteor threatened the city's safety, Thundertooth and his family used their unique abilities to divert it, saving the city from destruction. The citizens hailed them as heroes, and the toy factory became a symbol of resilience and unity.
```

---
**Mistral 7B:**
```
The story follows Thundertooth, a talking dinosaur who travels through time to the future city of New Eden. There, he meets Mayor Grace and her team of scientists, who help him find food without harming humans. The citizens of New Eden embrace Thundertooth as a symbol of unity between eras, and he becomes a beloved figure in the city.
 
One day, a massive meteor threatens to destroy the city. Thundertooth, along with his family Lumina, Echo, Sapphire, and Ignis, work together to divert the meteor's path using their unique talents. Their efforts are successful, and they save the city from destruction. The citizens hail them as heroes, and their legacy is forever etched in the city's history.
```

---
**Mixtral 8X7B:**
```
Thundertooth, a talking dinosaur from prehistoric times, finds himself in a futuristic city after passing through a strange portal. Initially overwhelmed by the unfamiliar sights and sounds, he struggles to find food without causing harm to the humans. The mayor of the city, Eleanor Grace, extends her hospitality and helps Thundertooth find a sustainable solution for his hunger.
 
As the story progresses, a meteor threatens Earth, and Thundertooth and his family step up to save the city. Lumina enhances the city's energy systems, Echo amplifies evacuation orders, Sapphire calms the panicked masses, and Ignis alters the meteor's trajectory. Their combined efforts successfully divert the catastrophic collision, ensuring the safety of the citizens. The Thundertooth family is hailed as heroes, and their legacy becomes a symbol of resilience and unity in the city.
```

---
**Neural Chat 7B:**
```
In a futuristic city, Thundertooth, a talking dinosaur, becomes an integral part of the community after being transported through time. He and his family establish a toy factory that produces magical widgets loved by both children and adults. When a meteor threatens to destroy their city, Thundertooth's family works together to save it using their unique abilities. Their heroic efforts unite the citizens, cementing the Thundertooth family as beloved heroes in the city's history.
```

---
**Orca 2:**
```
Key points:
- Thundertooth is a talking dinosaur who travels through time and ends up in a futuristic city
- He meets Mayor Grace and the citizens, who help him find food without harming anyone
- He starts a toy factory with his family that produces magical widgets
- He saves the city from a meteor threat with the help of his family's unique talents
 
Summary:
Thundertooth is a dinosaur who can talk and travel through time. He arrives in a futuristic city where he meets Mayor Grace, who welcomes him and helps him find food that does not harm anyone. He starts a toy factory with his family that makes amazing widgets that delight the people. When a meteor threatens to destroy the city, Thundertooth and his family use their abilities to divert it and save the day. They become heroes and symbols of unity in the city.
```
---
Phi-2:
```
In this story, Thundertooth, a dinosaur who traveled back in time to the year 2050, discovers a futuristic city where he meets his human counterpart, Seraphina. Together, they have four children with unique abilities inherited from their parents. When a meteor threatens the city, Thundertooth and his family use their talents to protect the citizens and divert the meteor's path. Their heroic efforts save the city, making them heroes in the eyes of its inhabitants. The toy factory becomes a symbol of unity and resilience, showcasing the power of cooperation between dinosaurs and humans.
```
---
**SOLAR 10.7B Instruct:**
```
In a futuristic city, residents discover an incoming meteor threatening their existence. Thundertooth, a unique dinosaur with human-like communication abilities, lives in this city and becomes the central figure in averting the disaster. Recognizing his importance, Mayor Grace enlists Thundertooth and his family to devise a plan to divert or neutralize the meteor.

Lumina, their eldest child, enhances the city's energy systems to create a protective force field. Echo, with his mimicry abilities, amplifies emergency signals for timely evacuation instructions. Sapphire, who possesses calming and healing powers, helps maintain order during the evacuation by comforting panicked citizens. Ignis, the youngest child, uses his fire-controlling talents to alter the meteor's trajectory and reduce its destructive force.

Thundertooth leads his family and the city's inhabitants in their combined efforts to deflect the meteor's deadly path. Their coordinated actions successfully divert the catastrophic collision, saving the city from destruction. The citizens celebrate Thundertooth and his family as true heroes, forever etching their legacy into the city's history.

Thundertooth, initially a stranger to this futuristic world, found a new home in the city after being transported through time. He embraced the advanced technology around him and founded a toy factory producing magical interactive widgets that captivated both children and adults. Thundertooth later married Seraphina, another intelligent dinosaur, and they had four unique children who inherited traits from their parents. Together, they became an integral part of the city's community, leaving a lasting impact through their family toy factory.
``` 

---
**Yi 34B:**

I could not get Yi-34B to complete inference, I tried lower quantized models but it still didn't complete. If anyone has Pi-34B working with LangChain and Ollama, help please!

---
#### Notes
- Getting the LLM to include citations with the sources is LLM dependent and heavily dependent on the prompt. It's important to get citations so I'll work on getting that for the #5 notebook.