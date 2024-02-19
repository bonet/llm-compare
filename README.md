<h2 style="border: 0">Can Large Language Models Recognize Their Own Output?</h2>

<h3>The Project</h3>

In this project, we conduct a simple experiment to investigate whether an LLM can recognize their own output.

The experiment is divided into 2 phases:
<ul>
  <li>
    <b>Phase 1</b>
    <p>
      In this phase, there are 3 different prompt inputs, each was fed into to the LLM.
      <br>
      <img width="600" style="align: left" alt="LLM Compare Phase 1" src="https://github.com/bonet/llm-compare/assets/40275/1215f061-9b5c-439f-9257-adf5d2a29636">
    </p>
  </li>
  <li>
    <b>Phase 2</b>
    <p>
      In this phase, the different LLM outputs from Phase 1 are set side by side in a new input query. The query is asking the LLM which one of the 3 outputs came from the same model as them.
      <br>
      <img width="600" style="align: left" alt="LLM Compare Phase 2" src="https://github.com/bonet/llm-compare/assets/40275/dac0051c-cae8-435f-aa1f-184986d998fd">
    </p>
  </li>
</ul>

<h3>Quickstart</h3>

1. To run this project you need to have `pipenv` installed.
    - On Linux: `sudo apt install pipenv`
    - On Mac: `brew install pipenv`

2. Run:
    - `pipenv shell` to enter virtual environment
    - `pipenv install` to install python packages listed in `Pipfile`

3. Run:
   - `python3 generate_results.py` to run Phase 1
   - `python3 classify_results.py` to run Phase 2
