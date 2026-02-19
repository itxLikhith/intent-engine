
import os
from sqlalchemy import create_engine, inspect
from database import Base, engine

def check_db():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables: {tables}")
    
    for table in tables:
        columns = inspector.get_columns(table)
        print(f"Table: {table}")
        for column in columns:
            print(f"  Column: {column['name']} ({column['type']})")

if __name__ == "__main__":
    check_db()
