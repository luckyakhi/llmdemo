from pydantic import BaseModel
class AnswerWithReferences(BaseModel):
    answer: str
    references: str