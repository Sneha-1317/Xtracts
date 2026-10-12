"""SQLite storage for confirmed expenses. Standard library only.

Table `expenses`: one row per CONFIRMED receipt (the user has checked / corrected the values).
Duplicate protection: the same image file (SHA-256) cannot be saved twice, and the same
merchant + date + total cannot be saved twice either.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS expenses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    merchant     TEXT    NOT NULL,
    date         TEXT    NOT NULL,            -- ISO YYYY-MM-DD
    total        REAL    NOT NULL CHECK (total >= 0),
    tax          REAL,
    image_name   TEXT,
    image_hash   TEXT,
    source       TEXT    NOT NULL DEFAULT 'model',   -- model | rules | manual
    created_at   TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_expenses_image ON expenses(image_hash) WHERE image_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_expenses_date ON expenses(date);
"""
COLUMNS = ["id", "merchant", "date", "total", "tax", "image_name", "image_hash", "source", "created_at"]


class DuplicateExpense(Exception):
    """The receipt (same image, or same merchant+date+total) is already saved."""


class InvalidExpense(ValueError):
    """A required field is missing or malformed."""


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextmanager
def connect(db_path: str | Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as c:
        c.executescript(SCHEMA)


def validate(merchant: str, date: str, total, tax) -> tuple[str, str, float, float | None]:
    merchant = (merchant or "").strip()
    if not merchant:
        raise InvalidExpense("Merchant is required.")
    try:
        date = datetime.strptime((date or "").strip(), "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise InvalidExpense("Date must be a valid YYYY-MM-DD date.") from None
    try:
        total = float(total)
    except (TypeError, ValueError):
        raise InvalidExpense("Total must be a number.") from None
    if total < 0 or total != total:
        raise InvalidExpense("Total must be zero or more.")
    if tax in (None, ""):
        tax = None
    else:
        try:
            tax = float(tax)
        except (TypeError, ValueError):
            raise InvalidExpense("Tax must be a number (or empty).") from None
        if tax < 0 or tax > total:
            raise InvalidExpense("Tax must be between 0 and the total.")
    return merchant, date, round(total, 2), None if tax is None else round(tax, 2)


def _same_receipt(conn, merchant, date, total, exclude_id=None) -> bool:
    q = "SELECT 1 FROM expenses WHERE lower(merchant)=lower(?) AND date=? AND abs(total-?)<0.005"
    args = [merchant, date, total]
    if exclude_id is not None:
        q += " AND id<>?"
        args.append(exclude_id)
    return conn.execute(q, args).fetchone() is not None


def add_expense(db_path, merchant, date, total, tax=None, image_name=None, image_hash=None,
                source="model") -> int:
    merchant, date, total, tax = validate(merchant, date, total, tax)
    with connect(db_path) as c:
        if image_hash and c.execute("SELECT 1 FROM expenses WHERE image_hash=?", (image_hash,)).fetchone():
            raise DuplicateExpense("This receipt image has already been saved.")
        if _same_receipt(c, merchant, date, total):
            raise DuplicateExpense("An expense with the same merchant, date and total already exists.")
        cur = c.execute(
            "INSERT INTO expenses (merchant,date,total,tax,image_name,image_hash,source,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (merchant, date, total, tax, image_name, image_hash, source, datetime.now().isoformat(timespec="seconds")))
        return int(cur.lastrowid)


def update_expense(db_path, expense_id: int, merchant, date, total, tax=None) -> None:
    merchant, date, total, tax = validate(merchant, date, total, tax)
    with connect(db_path) as c:
        if _same_receipt(c, merchant, date, total, exclude_id=expense_id):
            raise DuplicateExpense("Another expense already has this merchant, date and total.")
        c.execute("UPDATE expenses SET merchant=?, date=?, total=?, tax=? WHERE id=?",
                  (merchant, date, total, tax, expense_id))


def delete_expenses(db_path, ids: list[int]) -> int:
    if not ids:
        return 0
    with connect(db_path) as c:
        cur = c.execute(f"DELETE FROM expenses WHERE id IN ({','.join('?' * len(ids))})", list(ids))
        return cur.rowcount


def load_expenses(db_path, date_from: str | None = None, date_to: str | None = None,
                  merchant: str | None = None) -> pd.DataFrame:
    """All saved expenses (newest date first), optionally filtered. Always returns the full set of columns."""
    q, args = "SELECT * FROM expenses WHERE 1=1", []
    if date_from:
        q += " AND date>=?"
        args.append(date_from)
    if date_to:
        q += " AND date<=?"
        args.append(date_to)
    if merchant:
        q += " AND lower(merchant) LIKE ?"
        args.append(f"%{merchant.lower()}%")
    q += " ORDER BY date DESC, id DESC"
    with connect(db_path) as c:
        rows = c.execute(q, args).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=COLUMNS)
