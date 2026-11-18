from ..services import offset_studies
from ..utils.serialization import row


class OffsetStudiesController:
    list = staticmethod(offset_studies.list_studies)
    create = staticmethod(offset_studies.create_study)

    @staticmethod
    def get(db, id):
        return offset_studies.get_study(db, id)
