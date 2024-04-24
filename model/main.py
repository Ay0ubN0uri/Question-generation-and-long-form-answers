from datasets import load_dataset
from question_answer.pipelines import (
    get_preprocessing_pipeline, 
    get_question_answering_pipeline, 
    get_question_generation_pipeline, 
    connect_to_document_store
)


def preprocess(document_store):
    preprocessing_pipeline = get_preprocessing_pipeline(
        document_store,
        # use_ollama=True,
        use_openai=True
    )
    res = preprocessing_pipeline.run(
        {
            "file_type_router": {
                "sources": [
                    # "SDG.pdf",
                    # "articles.pdf",
                    "doc1.docx",
                    "doc2.docx",
                    # "index.html"
                ]
            }
        }
    )


def generate_questions(document_store):
    # model_name = "mistral-question-answering:latest"
    model_name = "llama2-question-answering:latest"
    question_generation_pipline = get_question_generation_pipeline(
        document_store,
        # use_openai=True
        ollama_model_name=model_name
    )
    res = question_generation_pipline.run({
        "retriever": {
            "filters": {}
        }
    })
    return res['question_generation_response']['questions']


def generate_answer(document_store, question):
    # model_name = "mistral-question-answering:latest"
    model_name = "llama2-question-answering:latest"
    question_answering_pipline = get_question_answering_pipeline(
        document_store,
        # question,
        # use_openai=True
        ollama_model_name=model_name
        # ollama_model_name='llama2-question-answering:latest'
    )
    response = question_answering_pipline.run(
        {"prompt_builder": {"query": question}, "retriever": {"query": question}})
    return print(response["llm"]["replies"][0])


def main():
    document_store = connect_to_document_store()
    print(
        f'number of documents in vector db : {document_store.count_documents()}')
    # document_store.delete_documents(
    #     [doc.id for doc in document_store.filter_documents()])
    # preprocess(document_store)
    print(
        f'number of documents in vector db : {document_store.count_documents()}')
    for doc in document_store.filter_documents():
        print(len(doc.content.split(' ')))
    questions = generate_questions(document_store)
    print(f'generated questions : {questions}')
    question = ''
    answer = generate_answer(document_store, question)
    print(f'answer for the question {question} : {answer}')


if __name__ == "__main__":
    main()
