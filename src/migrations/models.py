from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, MetaData, BigInteger, Boolean
from sqlalchemy.orm import relationship
from src.migrations.database import Base
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

metadata_obj = MetaData
logger.info(f'welcome models.py')

class detection_sugar_inclusions(Base):
    __tablename__ = "detection_sugar_inclusions"
    __table_args__ = {'extend_existing': True}
    metadata_obj,
    version = Column(BigInteger, primary_key=True, index=True)
    factory_id = Column(String(255))
    created = Column(DateTime)
    updated = Column(DateTime, default=datetime.utcnow)
    trash = Column(Integer)
    not_trash = Column(Integer)
    arm = Column(Integer)

