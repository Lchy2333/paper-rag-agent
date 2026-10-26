"""query_usage 仓储单测：锁定 pool 的 row_factory=dict_row（返回字典而非元组）。"""
from app import db


class _Cursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [self._row] if self._row is not None else []


class _Conn:
    def __init__(self, row):
        self._row = row
        self._executed = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        self._executed = (sql, params)
        return _Cursor(self._row)


class _Pool:
    def __init__(self, row):
        self._row = row
        self.conn = _Conn(row)

    def connection(self):
        return self.conn


def test_get_session_usage_parses_dict_row():
    """main.py 的 pool 用 dict_row：fetchone 返回字典，必须按列名取值。

    聚合列必须带 AS 别名——不带时 psycopg 以函数名 coalesce 作列名，
    两个同名列还会互相覆盖，dict_row 里找不到 input_tokens/output_tokens。
    """
    pool = _Pool({"input_tokens": 43, "output_tokens": 25})
    assert db.get_session_usage(pool, "t1") == (43, 25)
    sql, _ = pool.conn._executed
    assert "AS input_tokens" in sql and "AS output_tokens" in sql


def test_get_session_usage_empty_returns_zero():
    """无历史记录（fetchone 返回 None）→ (0, 0)。"""
    pool = _Pool(None)
    assert db.get_session_usage(pool, "t1") == (0, 0)


def test_record_usage_issues_insert():
    """record_usage 写入一行（thread_id / user_id / 本次用量 / 上下文水位）。"""
    pool = _Pool(None)
    db.record_usage(pool, "t1", "u1", 15, 5, 1, 2979)
    sql, params = pool.conn._executed
    assert "INSERT INTO query_usage" in sql
    assert "last_input_tokens" in sql
    assert params == ("t1", "u1", 15, 5, 1, 2979)


class _SeqConn:
    """按执行顺序返回不同行：get_thread_usage 会连发 SUM 与"最近一行"两条 SQL。"""

    def __init__(self, rows):
        self._rows = list(rows)
        self.sqls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        self.sqls.append(sql)
        return _Cursor(self._rows.pop(0))


class _SeqPool:
    def __init__(self, rows):
        self.conn = _SeqConn(rows)

    def connection(self):
        return self.conn


def test_get_thread_usage_combines_sum_and_last_input():
    """累计用量取 SUM，水位取最近一行 last_input_tokens（不累加）。"""
    pool = _SeqPool([
        {"input_tokens": 4241, "output_tokens": 444},
        {"last_input_tokens": 2979},
    ])
    assert db.get_thread_usage(pool, "t1") == {
        "input_tokens": 4241,
        "output_tokens": 444,
        "last_input_tokens": 2979,
    }
    # 水位必须是"最近一行"，否则恢复出来的环会对应旧轮次
    assert "ORDER BY created_at DESC, id DESC LIMIT 1" in pool.conn.sqls[1]


def test_get_thread_usage_no_history_returns_zero():
    pool = _SeqPool([None, None])
    assert db.get_thread_usage(pool, "t1") == {
        "input_tokens": 0,
        "output_tokens": 0,
        "last_input_tokens": 0,
    }
