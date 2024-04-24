# Question Generation & Long Form Question Aswering RAG

## Project Workflow
![Project Workflow](images/project%20workflow.jpg)

## Evaluation Methodology
The journey of evaluating an LLM application ideally follows a structured framework, incorporating a suite of specialized tools and libraries. By systematically applying evaluation methods, we can gain meaningful insights into our applications, ensuring they meet our standards and deliver the desired outcomes.

![RAG Evaluation Workflow](images/rag%20evaluation%20workflow.png)

## LLMs Throughput
![LLMs Throughput](images/llm_throughput.png)

## LLMs Evaluation
![LLMs Evaluation](images/llm_evaluation.png)

1. **Faithfulness** : Measures the factual consistency of the answer to the context based on the question.

2. **Context_precision** : Measures how relevant the retrieved context is to the question, conveying the quality of the retrieval pipeline.

3. **Answer_relevancy** : Measures how relevant the answer is to the question.

4. **Context_recall** : Measures the retriever’s ability to retrieve all necessary information required to answer the question.

## How to install

1. Create a virtual env and install packages.
    ```bash
    python3 -m venv .venv
    pip install -r requirements.txt
    ```

2. Install ollama from [here](https://ollama.com/download).
3. Run weaviate vector database using docker compose file.
    ```bash
    docker compose up -d
    ```

4. Fine-tune your llm model using **Modelfile**.
    ```bash
    ollama create question-answering -f Modelfile
    ```
5. Test the model.

## NPC Avatar