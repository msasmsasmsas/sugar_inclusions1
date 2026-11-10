# dto.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime

def to_camel_case(string: str) -> str:
    parts = string.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

class detectionSugarInclusions(BaseModel):
    version: Optional[str] = None
    factory_id: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    trash: Optional[str] = None
    not_trash: Optional[str] = None
    arm: Optional[str] = None


    class Config:
        orm_mode = True
        alias_generator = to_camel_case
        allow_population_by_field_name = True




