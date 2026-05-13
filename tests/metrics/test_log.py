"""MetricSQL roundtrip: locks in §1.7.

The pre-fix INSERT referenced a ``resume`` column that the CREATE TABLE
didn't define; the first ``log()`` call raised an OperationalError.
"""

import sqlite3

from src.utils.metrics.log.metric_sql import MetricSQL


def test_log_does_not_raise(tmp_path):
    db = tmp_path / "report.db"
    sql = MetricSQL(str(db))
    sql.log(_epoch=1, _step=10, _seen_labels=80, _tag='train',
            _metrics={'Loss': 0.5, 'Dice': 0.42})


def test_log_persists_rows(tmp_path):
    db = tmp_path / "report.db"
    sql = MetricSQL(str(db))
    sql.log(_epoch=0, _step=1, _seen_labels=8, _tag='train',
            _metrics={'Loss': 1.0, 'Dice': 0.0})
    sql.log(_epoch=0, _step=2, _seen_labels=16, _tag='valid',
            _metrics={'Loss': 0.9, 'Dice': 0.1})

    with sqlite3.connect(db) as con:
        rows = con.execute(
            "SELECT epoch, step, lseen, tag, name, value FROM metrics ORDER BY id"
        ).fetchall()
    assert len(rows) == 4
    assert rows[0] == (0, 1, 8, 'train', 'Loss', 1.0)
    assert rows[3] == (0, 2, 16, 'valid', 'Dice', 0.1)


def test_insert_column_list_matches_schema(tmp_path):
    """Guard against the original bug: INSERT column list must be a subset of the CREATE TABLE columns."""
    db = tmp_path / "report.db"
    MetricSQL(str(db))  # creates the table
    with sqlite3.connect(db) as con:
        cols = {row[1] for row in con.execute("PRAGMA table_info(metrics)").fetchall()}
    # The columns the INSERT names, verbatim
    insert_cols = {'epoch', 'step', 'lseen', 'tag', 'name', 'value'}
    assert insert_cols.issubset(cols), (
        f"INSERT references columns not in schema: {insert_cols - cols}")
