import streamlit as st
from question_answer.pipelines import (
    connect_to_document_store,
    get_preprocessing_pipeline,
    get_question_generation_pipeline,
    get_question_answering_pipeline
)
import tempfile
from pathlib import Path


@st.cache_data
def process_uploaded_files(file_paths, _preprocessing_pipeline, _question_generation_pipeline):
    res = _preprocessing_pipeline.run(
        {
            "file_type_router": {
                "sources": file_paths
            }
        }
    )
    print("+++++++++ result ")
    print(res)
    res = _question_generation_pipeline.run({
        "retriever": {
            "filters": {}
        }
    })
    st.session_state.question_generated = True
    return res['question_generation_response']['questions']
    # return ['How does supervised learning differ from unsupervised learning in terms of the type of data used for training and the presence of predefined labels?',
    #         'Can you explain the key differences between classification and regression tasks in supervised learning, along with examples of each?',
    #         'What are some common techniques used in unsupervised learning to identify hidden patterns or structures within unlabeled data?',
    #         'How does reinforcement learning differ from supervised and unsupervised learning, specifically in terms of how the model learns and improves its decision-making processes?',
    #         'What are some of the notable successes of deep learning in various fields, and how does it differ from traditional machine learning approaches?',
    #         'How has the adoption of AI and machine learning impacted industries such as healthcare, finance, transportation, and entertainment, leading to innovations in specific applications?',
    #         'What advantages does Python offer as a programming language, especially in terms of its design philosophy, versatility in programming paradigms, and ecosystem of libraries?',
    #         "How has Python's simplicity, readability, and extensive standard library contributed to its widespread adoption across different domains, including web development and scientific computing?",
    #         'What role does data quality and quantity play in influencing the performance of machine learning models, and what are the critical steps in the machine learning pipeline to enhance model outcomes?',
    #         'Can you elaborate on how machine learning algorithms generalize knowledge from training data to make predictions on new data, and the impact of reinforcement learning in domains such as gameplaying, robotics, and autonomous driving?']


@st.cache_resource
def init_model():
    document_store = connect_to_document_store()
    preprocessing_pipeline = get_preprocessing_pipeline(
        document_store, use_openai=True)
    question_generation_pipeline = get_question_generation_pipeline(
        document_store=document_store, use_openai=True)
    question_answering_pipeline = get_question_answering_pipeline(
        document_store=document_store, use_openai=True)
    return document_store, preprocessing_pipeline, question_generation_pipeline, question_answering_pipeline


def main():
    submitted = False
    st.set_page_config(
        layout="wide", page_title="Question Generation, and Long Form Question Answering", page_icon='😎')
    if 'question_generated' not in st.session_state:
        st.session_state.question_generated = False
    document_store, preprocessing_pipeline, question_generation_pipeline, question_answering_pipeline = init_model()
    print(document_store.count_documents())
    st.title(
        "Question :blue[Generation], and Long Form Question :red[Answering] :sunglasses:")
    # st.write(st.session_state.question_generated)
    uploaded_files = st.file_uploader("Upload a pdf file", type=[
        'pdf', 'txt', 'docx', 'html'], accept_multiple_files=True)
    if len(uploaded_files) == 0:
        print("No uploaded file")
        st.session_state.question_generated = False
    else:
        print("there is some uploaded file")
        file_paths = []
        for uploaded_file in uploaded_files:
            # print(uploaded_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = Path(tmp_file.name)

            uploaded_file_path = Path(tmp_file_path)
            file_paths.append(uploaded_file_path)

            # st.write("Uploaded file saved to:", uploaded_file_path)
        print(f"============= file paths count : {len(file_paths)}")
        print(file_paths)
        questions = []
        if st.session_state.question_generated == False:
            print("99999999999 before process_uploaded_files line")
            questions = process_uploaded_files(
                file_paths, preprocessing_pipeline, question_generation_pipeline)
            st.session_state.questions = questions
        else:
            questions = st.session_state.questions

        left_column, right_column = st.columns(2)
        st.write(" ")
        result = st.container()
        with left_column:
            st.title("Input question")
            question_input = st.text_input("Ask a question")
            if question_input:
                if st.button("Submit", key='input'):
                    with result:
                        with st.expander(f"Question : _{question_input}_"):
                            response = question_answering_pipeline.run(
                                {"prompt_builder": {"query": question_input}, "retriever": {"query": question_input}})
                            st.write(response["llm"]["replies"][0])

        with right_column:
            st.title("Generated question")
            selected_question = st.radio(
                "Generated questions", questions, index=None)
            if selected_question:
                if st.button("Submit", key='select'):
                    with result:
                        with st.expander(f"Question : _{selected_question}_"):
                            response = question_answering_pipeline.run(
                                {"prompt_builder": {"query": selected_question}, "retriever": {"query": selected_question}})
                            st.write(response["llm"]["replies"][0])


if __name__ == "__main__":
    main()
