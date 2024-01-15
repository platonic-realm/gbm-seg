# Python Imports

# Library Imports
import sqlite3

# Local Imports


class MetricSQL():

    def __init__(self, _database_path: str):
        self.database_path = _database_path

        with sqlite3.connect(self.database_path) as con:
            cursor = con.cursor()

        create_table = '''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER NOT NULL,
                step INTEGER NOT NULL,
                lseen INTEGER NOT NULL,
                tag TEXT NOT NULL,
                name TEXT NOT NULL,
                value REAL)
                       '''
        cursor.execute(create_table)

        con.commit()

    def log(self,
            _epoch,
            _step,
            _seen_labels,
            _tag,
            _metrics) -> None:

        with sqlite3.connect(self.database_path) as con:
            cursor = con.cursor()

            insert_query = '''
                INSERT INTO metrics (resume, epoch, step,
                                     lseen, tag, name, value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                           '''

            values = [(_epoch,
                       _step,
                       _seen_labels,
                       _tag,
                       name,
                       float(value)) for name, value in _metrics.items()]

            cursor.executemany(insert_query, values)
            con.commit()
