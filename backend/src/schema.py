from pydantic import BaseModel
from typing import List


class Query(BaseModel):
    documents: str
    questions: List[str]


class DocParser(BaseModel):
    content: str
    document_type: str
    status: str


class Answer(BaseModel):
    answers: List[str]