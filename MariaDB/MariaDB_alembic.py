import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import logging

# Define the MariaDB engine using MariaDB Connector/Python
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
use_db_connection = False
if use_db_connection:
   try:
      #engine = sqlalchemy.create_engine("mariadb+mariadbconnector://root:Gw8yXCZQgn@10.43.32.125:3306/sugar_inclusions_ai")
      engine = sqlalchemy.create_engine("mariadb+mariadbconnector://root:PatUzzNt4A@10.47.55.177:3306/sugar_inclusions_ai")

      Base = declarative_base()

      class clDetection_sugar_inclusions(Base):
         __tablename__ = 'detection_sugar_inclusions'
         version = sqlalchemy.Column(sqlalchemy.BigInteger, primary_key=True, nullable=False)
         created = sqlalchemy.Column(sqlalchemy.DateTime)
         updated = sqlalchemy.Column(sqlalchemy.DateTime)
         trash = sqlalchemy.Column(sqlalchemy.SMALLINT, nullable=True)
         not_trash = sqlalchemy.Column(sqlalchemy.SMALLINT, nullable=True)

      Base.metadata.create_all(engine)
   except OSError as e:
      logger.error("MariaDB: Ошибка миграции БД: %s" % (e.strerror))

# Create a session
#Session = sqlalchemy.orm.sessionmaker()
#Session.configure(bind=engine)
#session = Session()

def add_detection_sugar_inclusions(session,cur_datetime,trash,not_trash):
   cur_datetime=cur_datetime[0:19]
   _created = datetime.strptime(cur_datetime, "%Y-%m-%d %H:%M:%S")

   _version=(_created.timestamp())
   newdetection_sugar_inclusions = clDetection_sugar_inclusions(version=_version,created=_created,updated=_created,trash=trash, not_trash=not_trash)
   session.add(newdetection_sugar_inclusions)
   session.commit()

def selectAll(session):
   detection_sugar_inclusions = session.query(clDetection_sugar_inclusions).all()
   for detection_sugar_inclusion in detection_sugar_inclusions:
       print(detection_sugar_inclusion.created+" ::: " + detection_sugar_inclusion.trash + ' ' + detection_sugar_inclusion.not_trash)

def selectByStatus(session,trash):
   detection_sugar_inclusions = session.query(clDetection_sugar_inclusions).filter_by(trash>0)
   for detection_sugar_inclusion in detection_sugar_inclusions:
       print(detection_sugar_inclusion.created+" ::: " + detection_sugar_inclusion.trash + ' ' + detection_sugar_inclusion.not_trash)

def update_detection_sugar_inclusions_Status(session,created):
   detection_sugar_inclusions = session.query(clDetection_sugar_inclusions).get(created)
   detection_sugar_inclusions.updated = datetime.now()
   session.commit()

def delete_detection_sugar_inclusions(session,created):
   session.query(clDetection_sugar_inclusions).filter(clDetection_sugar_inclusions.created == created).delete()
   session.commit()

# Add some new employees
#addEmployee("Bruce", "Wayne")
#addEmployee("Diana", "Prince")
#addEmployee("Clark", "Kent")

# Show all employees
#print('All Employees')
#selectAll()
#print("----------------")

# Update employee status
#updateEmployeeStatus(2,False)

# Show active employees
#print('Active Employees')
#selectByStatus(True)
#print("----------------")

# Delete employee
#deleteEmployee(1)

# Show all employees
#print('All Employees')
#selectAll()
#print("----------------")