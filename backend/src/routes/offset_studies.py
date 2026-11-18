from fastapi import APIRouter, Depends

from ..config.database import db_session
from ..schemas.offset_study import OffsetStudyInput
from ..services.auth import current_user, writer
from ..controllers.offset_studies import OffsetStudiesController as C

router = APIRouter()


@router.get('/offset-studies')
def studies(db=Depends(db_session), u=Depends(current_user)):
    return C.list(db)


@router.post('/offset-studies')
def create(data: OffsetStudyInput, db=Depends(db_session), u=Depends(writer)):
    return C.create(db, u, data.run_id, data)


@router.get('/offset-studies/{id}')
def detail(id: int, db=Depends(db_session), u=Depends(current_user)):
    return C.get(db, id)
