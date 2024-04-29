from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import uvicorn
from question_answer.pipelines import (
    connect_to_document_store,
    get_preprocessing_pipeline,
    get_question_generation_pipeline,
    get_question_answering_pipeline
)
import tempfile
from pathlib import Path

app = FastAPI()

ALLOWED_EXTENSIONS = {'docx', 'html', 'pdf', 'txt'}
UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    gen_questions: bool = False,
    num_questions: int = 10,
    use_openai: bool = True,
    model_name: str = 'mistral:latest'
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    file_paths = []
    for file in files:
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400, detail="Only docx, html, pdf, and txt files are allowed")

        # print(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = Path(tmp_file.name)

            uploaded_file_path = Path(tmp_file_path)
            file_paths.append(uploaded_file_path)
    # print(file_paths)
    preprocessing_pipeline = get_preprocessing_pipeline(
        document_store, use_openai=True)
    if use_openai:
        question_generation_pipeline = get_question_generation_pipeline(
            document_store=document_store, use_openai=True, num=num_questions)
    else:
        question_generation_pipeline = get_question_generation_pipeline(
            document_store=document_store, num=num_questions, ollama_model_name=model_name)
    res = preprocessing_pipeline.run(
        {
            "file_type_router": {
                "sources": file_paths
            }
        }
    )
    print("+++++++++ result ")
    print(res)
    if gen_questions:
        res = question_generation_pipeline.run({
            "retriever": {
                "filters": {}
            }
        })
        return {
            "message": "Files uploaded successfully",
            "questions": res['question_generation_response']['questions']
        }
    else:
        return {
            "message": "Files uploaded successfully",
        }


class Question(BaseModel):
    question: str


@app.post("/generate_answer")
async def generate_answer(
    question: Question,
    use_openai: bool = True,
    model_name: str = "mistral:latest"
):
    received_question = question.question
    if use_openai:
        question_answering_pipeline = get_question_answering_pipeline(
            document_store=document_store, use_openai=True)
    else:
        question_answering_pipeline = get_question_answering_pipeline(
            document_store=document_store, ollama_model_name=model_name)
    response = question_answering_pipeline.run(
        {"prompt_builder": {"query": received_question}, "retriever": {"query": received_question}})
    return {
        "question": received_question,
        "answer": response["llm"]["replies"][0]
    }


@app.get("/")
async def index():
    return {
        "hello": "Make with ❤️ V0rtex 42"
    }

document_store = connect_to_document_store()


def main():
    config = uvicorn.Config(
        app=app, host='0.0.0.0', port=8000)
    server = uvicorn.Server(config=config)
    server.run()


if __name__ == "__main__":
    main()
