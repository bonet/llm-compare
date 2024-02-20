<h2 style="border: 0">LLM Output Classifier</h2>

<h3>The Project</h3>

In this project, we conduct a simple experiment to investigate whether an LLM can recognize their own output.

The experiment is divided into 2 phases:
<ul>
  <li>
    <b>Phase 1</b>
    <p>
      In this phase, we feed prompt inputs to 3 different LLMs (<b>OpenAI GPT4</b>, <b>Llama 2</b>, and <b>Mixtral 8x7B</b>) and save the results into files.
      <br>
      <img width="600" style="align: left" alt="LLM Compare Phase 1" src="https://github.com/bonet/llm-compare/assets/40275/1215f061-9b5c-439f-9257-adf5d2a29636">
    </p>
  </li>
  <li>
    <b>Phase 2</b>
    <p>
      We build a classification query that is asking each LLM to determine which one of the outputs from Phase 1 are theirs.
      <br>
      <img width="600" style="align: left" alt="LLM Compare Phase 2" src="https://github.com/bonet/llm-compare/assets/40275/dac0051c-cae8-435f-aa1f-184986d998fd">
    </p>
  </li>
</ul>

<h3>LLM wrapper</h3>

[LLMCaller](llm_caller.py) is a wrapper for calling various LLMs dynamically. Under the hood, the class is using Ollama platform to locally connect with Llama 2 and Mixtral models and Langchain to remotely connect with OpenAI GPT-4.

<h3>Quickstart</h3>

1. To run this project you need to have `pipenv` installed.
    - On Linux: `sudo apt install pipenv`
    - On Mac: `brew install pipenv`

2. If you don't have Ollama already, you can download it [here](https://ollama.com/download)

3. Add environment variables:
    - `cp .env.example .env` to copy environment file
    - Add your OpenAI API key to `.env` file

4. Install packages:
    - `pipenv shell` to enter virtual environment
    - `pipenv install` to install python packages listed in `Pipfile`

5. Run Ollama server:
   - `ollama serve` to run Ollama server locally and handle Llama 2 and Mixtral prompt request

6. On another shell, run the script:
   - `python3 generate_results.py` to run Phase 1
   - `python3 classify_results.py` to run Phase 2

7. The classification result can be found [here](results/classification_results.txt)
