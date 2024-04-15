# preprocessing
from typing import List, Optional
from haystack.components.retrievers import FilterRetriever
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.converters import PyPDFToDocument, TextFileToDocument, HTMLToDocument, TikaDocumentConverter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.components.joiners import DocumentJoiner
from haystack import Document, Pipeline
from sentence_transformers import SentenceTransformer
from haystack.utils import Secret
from haystack.components.builders.prompt_builder import PromptBuilder
# document store
from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
# Retrievers
from haystack_integrations.components.retrievers.weaviate.bm25_retriever import WeaviateBM25Retriever
from haystack_integrations.components.retrievers.weaviate.embedding_retriever import WeaviateEmbeddingRetriever
# LLMs
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.generators import OpenAIGenerator
# Embedders
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import OpenAIDocumentEmbedder
import os
import mimetypes
mimetypes.init()


from haystack import component
import re


@component
class QuestionGenerationResponse:
    @component.output_types(questions=List[str])
    def run(self, replies: List[str]):
        questions = []
        for idx, response in enumerate(replies):
            result = str(response).strip().split("\n")
            cleaned_questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            questions += [
                question for question in cleaned_questions if len(question) > 0
            ]
        return {
            'questions': questions
        }



def connect_to_document_store():
    document_store = WeaviateDocumentStore(url='http://localhost:8080')
    document_store._client.batch.configure(
        batch_size=100,
        dynamic=True
    )
    return document_store


def get_preprocessing_pipeline(
    document_store, 
    use_openai=False, 
    use_ollama=False
):
    file_type_router = FileTypeRouter(
        mime_types=[
            "application/pdf",
            "text/plain",
            "text/html",
            # this for doc file ext
            # "application/msword",
            # this for docx file ext
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ])
    pdf_converter = PyPDFToDocument()
    text_file_converter = TextFileToDocument()
    html_file_converter = HTMLToDocument("ArticleExtractor")
    tika_file_converter = TikaDocumentConverter()
    document_cleaner = DocumentCleaner()
    document_joiner = DocumentJoiner()
    document_splitter = DocumentSplitter(
        split_by="word", split_length=150, split_overlap=50)
    document_writer = DocumentWriter(document_store)
    document_embedder = None
    if use_openai:
        document_embedder = OpenAIDocumentEmbedder(
            api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")),
            meta_fields_to_embed=["title"],
            dimensions=784,
            model='text-embedding-3-large'
        )
    elif use_ollama:
        document_embedder = OllamaDocumentEmbedder(
            model="nomic-embed-text", url="http://localhost:11434/api/embeddings")
    else:
        document_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2")

    preprocessing_pipeline = Pipeline()
    # add components
    preprocessing_pipeline.add_component(
        instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(
        instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(
        instance=text_file_converter, name="text_file_converter")
    preprocessing_pipeline.add_component(
        instance=html_file_converter, name="html_file_converter")
    preprocessing_pipeline.add_component(
        instance=tika_file_converter, name="tika_file_converter")
    preprocessing_pipeline.add_component(
        instance=document_joiner, name="document_joiner")
    preprocessing_pipeline.add_component(
        instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(
        instance=document_splitter, name="document_splitter")
    preprocessing_pipeline.add_component(
        instance=document_embedder, name="document_embedder")
    preprocessing_pipeline.add_component(
        instance=document_writer, name="document_writer")
    # connect them
    preprocessing_pipeline.connect(
        "file_type_router.application/pdf", "pypdf_converter.sources")
    preprocessing_pipeline.connect(
        "file_type_router.text/plain", "text_file_converter.sources")
    preprocessing_pipeline.connect(
        "file_type_router.text/html", "html_file_converter.sources")
    # preprocessing_pipeline.connect(
    #     "file_type_router.application/msword", "tika_file_converter.sources")
    preprocessing_pipeline.connect(
        "file_type_router.application/vnd.openxmlformats-officedocument.wordprocessingml.document", "tika_file_converter.sources")
    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("text_file_converter", "document_joiner")
    preprocessing_pipeline.connect("html_file_converter", "document_joiner")
    preprocessing_pipeline.connect("tika_file_converter", "document_joiner")
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")
    return preprocessing_pipeline


def get_question_generation_pipeline(
    document_store, 
    num: int = 10, 
    use_openai=False, 
    ollama_model_name: str = 'mistral-question-answering:latest'
):
    system_prompt = f"You are proficient in generating questions aimed at fostering comprehensive understanding across various aspects of the provided information. Your task is to produce {num} questions that prompt detailed responses, spanning different areas of the context. Your responses should consist solely of questions, designed to elicit comprehensive long-form answers. Avoid injecting personal knowledge; rely exclusively on the given context. If uncertain or lacking sufficient details, refrain from formulating questions. Your objective is to facilitate deep exploration and understanding through a diverse range of thought-provoking inquiries."
    llm = None
    if use_openai:
        llm = OpenAIGenerator(
            api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")),
            system_prompt=system_prompt
        )
    else:
        llm = OllamaGenerator(
            model=ollama_model_name,
            url="http://localhost:11434/api/generate",
            timeout=3600,
            system_prompt=system_prompt,
            # streaming_callback=lambda x: print(x.content)
        )
    template = """
    Given the following information, generate questions.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    """
    question_generation_pipline = Pipeline()

    question_generation_pipline.add_component(
        "retriever", FilterRetriever(document_store=document_store))
    question_generation_pipline.add_component(
        "prompt_builder", PromptBuilder(template=template))
    question_generation_pipline.add_component("llm", llm)
    question_generation_pipline.add_component(
        "question_generation_response", QuestionGenerationResponse())
    question_generation_pipline.connect(
        "retriever", "prompt_builder.documents")
    question_generation_pipline.connect("prompt_builder", "llm")
    question_generation_pipline.connect(
        "llm.replies", "question_generation_response")
    return question_generation_pipline


def get_question_answering_pipeline(
    document_store, 
    use_openai=False, 
    ollama_model_name: str = 'mistral-question-answering:latest'
):
    template = """
    Given only the following information, answer the question.
    Ignore your own knowledge.
    

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{ query }}?
    """
    system_prompt = """
        You are an expert in question answering but based on the provided informations.
        You don't give answers from your own knowledge.If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Again and again don't give answers from your own knowledge.If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Try to answer from the provided context not from your knowledge.
        and don't say i don't know and then say but however....
    """
    llm = None
    if use_openai:
        llm = OpenAIGenerator(
            api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")),
            system_prompt=system_prompt
        )
    else:
        llm = OllamaGenerator(
            model=ollama_model_name,
            url="http://localhost:11434/api/generate",
            timeout=3600,
            # system_prompt=system_prompt,
            # streaming_callback=lambda x: print(x.content)
        )
    question_answering_pipline = Pipeline()
    question_answering_pipline.add_component(
        "retriever", WeaviateBM25Retriever(document_store=document_store))
    question_answering_pipline.add_component(
        "prompt_builder", PromptBuilder(template=template))
    question_answering_pipline.add_component("llm", llm)
    question_answering_pipline.connect("retriever", "prompt_builder.documents")
    question_answering_pipline.connect("prompt_builder", "llm")
    return question_answering_pipline

def get_question_answering_pipeline_for_evaluation(
    use_openai=False, 
    ollama_model_name: str = 'mistral-question-answering:latest'
):
    template = """
    Given only the following information, answer the question.
    Ignore your own knowledge.
    

    Context:
    {{ context }}

    Question: {{ query }}?
    """
    system_prompt = """
        You are an expert in question answering but based on the provided informations.
        You don't give answers from your own knowledge.If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Again and again don't give answers from your own knowledge.If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Try to answer from the provided context not from your knowledge.
        and don't say i don't know and then say but however....
    """
    llm = None
    if use_openai:
        llm = OpenAIGenerator(
            api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")),
            system_prompt=system_prompt
        )
    else:
        llm = OllamaGenerator(
            model=ollama_model_name,
            url="http://localhost:11434/api/generate",
            timeout=3600,
            # system_prompt=system_prompt,
            # streaming_callback=lambda x: print(x.content)
        )
    question_answering_pipline = Pipeline()
    # question_answering_pipline.add_component(
    #     "retriever", WeaviateBM25Retriever(document_store=document_store))
    question_answering_pipline.add_component(
        "prompt_builder", PromptBuilder(template=template))
    question_answering_pipline.add_component("llm", llm)
    # question_answering_pipline.connect("retriever", "prompt_builder.documents")
    question_answering_pipline.connect("prompt_builder", "llm")
    return question_answering_pipline