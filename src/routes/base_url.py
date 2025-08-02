from fastapi import APIRouter, Depends
from auth import auth
from utils import runner
import schema


router = APIRouter(prefix="/api/v1", tags=["Document Processor"])

@router.post("/hackrx/run")
def run_app(query: schema.Query, token: str = Depends(auth.verify_token)):
    
    llm_reponse = runner.executeAll(query)

    return {"answers": llm_reponse}
