"""Rule-based ("weak") labels for Merchant, Date, Total and Tax from OCR words.

These are SUGGESTIONS to speed up annotation, not ground truth. They will be wrong on
some receipts; a human verifies them in the review app. Receipts where the rules were
unsure get `flags` so they can be reviewed first.

Everything works on the OCR record produced by Step 2:
    {"width", "height", "words": [{"text", "box": [x0, y0, x1, y1], "conf", "line_id"}, ...]}
"""
from __future__ import annotations

import re
from statistics import median

# ---------------------------------------------------------------- rows
def group_rows(words: list[dict]) -> list[list[int]]:
    """Group word indices into visual rows (top-to-bottom, left-to-right inside a row)."""
    order = sorted(range(len(words)), key=lambda i: (words[i]["box"][1] + words[i]["box"][3]) / 2)
    rows: list[list[int]] = []
    for i in order:
        b = words[i]["box"]
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        if rows:
            last = rows[-1]
            row_cy = sum((words[j]["box"][1] + words[j]["box"][3]) / 2 for j in last) / len(last)
            row_h = median(words[j]["box"][3] - words[j]["box"][1] for j in last)
            if abs(cy - row_cy) <= 0.5 * min(h, row_h):
                last.append(i)
                continue
        rows.append([i])
    return [sorted(r, key=lambda j: words[j]["box"][0]) for r in rows]


def _row_text(words, row) -> str:
    return " ".join(words[i]["text"] for i in row)


# ---------------------------------------------------------------- amounts
_AMOUNT = re.compile(r"^\D{0,4}(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2}|\d+,\d{2}|\.\d{2})\D{0,2}$")
# loose: also 1 decimal digit ("1.4"). Only used inside a GST summary table, where the column is certain.
_AMOUNT_LOOSE = re.compile(r"^\D{0,4}(\d{1,3}(?:,\d{3})*\.\d{1,2}|\d+\.\d{1,2}|\d+,\d{2}|\.\d{2})\D{0,2}$")
_O_AS_ZERO = re.compile(r"(?<=RM)[Oo](?=[.\d])|(?<=\d)[Oo](?=\d|\.|$)|(?<=[.,])[Oo]")


def parse_amount(text: str, loose: bool = False) -> float | None:
    """'RM1,234.50' -> 1234.5 ; '525.00*' -> 525.0 ; '.65' -> 0.65 ; 'RMO.47' -> 0.47 ; 'abc' -> None.
    Needs 2 decimals (1 allowed when loose=True)."""
    t = _O_AS_ZERO.sub("0", text.strip())
    m = (_AMOUNT_LOOSE if loose else _AMOUNT).match(t)
    if not m:
        return None
    s = m.group(1)
    if "." not in s and "," in s:      # European decimal comma: 12,50
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _amounts_in_row(words, row) -> list[tuple[int, float]]:
    return [(i, v) for i in row if (v := parse_amount(words[i]["text"])) is not None]


# ---------------------------------------------------------------- date
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_DATE_PATTERNS = [
    re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$"),
    re.compile(r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}$"),
    re.compile(rf"^\d{{1,2}}[ \-/.]?{_MON}[ \-/,.]*\d{{2,4}}$", re.I),
    re.compile(rf"^{_MON}\.?[ \-]\d{{1,2}},?[ \-]\d{{2,4}}$", re.I),
]


def _valid_numeric_date(s: str) -> bool:
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if len(nums) != 3:
        return True  # month-name formats
    a, b, c = nums
    if len(str(re.findall(r"\d+", s)[0])) == 4:          # yyyy-mm-dd
        return 1990 <= a <= 2100 and 1 <= b <= 12 and 1 <= c <= 31
    year_ok = (1990 <= c <= 2100) if c > 99 else True
    return year_ok and 1 <= a <= 31 and 1 <= b <= 31 and (a <= 12 or b <= 12)


def _is_date(text: str) -> bool:
    t = text.strip(" ,;:()")
    return any(p.match(t) for p in _DATE_PATTERNS) and _valid_numeric_date(t)


# OCR often glues a date to its neighbours: "Date:05/02/2018", "19-03-201818:08:38" (date+time),
# "#19536001/06/2018" (invoice number + date). So also look for a date INSIDE a word.
_EMBED_DMY = re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.]((?:19|20)\d{2}|\d{2}(?=\d{1,2}:\d{2}|\D|$))(?![/\-.]\d)")
_EMBED_YMD = re.compile(r"((?:19|20)\d{2})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)(?![/\-.]\d)")
_EMBED_COMPACT = re.compile(r"(?<!\d)(20[0-3]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")  # 20180428
_EMBED_NAME = re.compile(rf"(?<!\d)\d{{1,2}}[ \-/.]?{_MON}[ \-/,.]*(?:19|20)?\d{{2}}", re.I)
_LABEL_BEFORE_DATE = ("date", "dt", "tarikh")


def _scan(pattern: re.Pattern, text: str):
    """Every match starting at ANY position, including ones overlapping an earlier (invalid) match.
    (re.finditer skips them: in '#19-18/01/2018' it would use up '19-18/01' and never see '18/01/2018'.)"""
    for pos in range(len(text)):
        m = pattern.match(text, pos)
        if m:
            yield m


def _label_ok(text: str, pos: int) -> bool:
    """Letters right before a date must be a 'Date' label. This rejects product codes such as
    WA10-24/72 or GB12-21-22-36; digits and punctuation before the date are fine (#1953601/06/2018)."""
    j = pos
    while j > 0 and text[j - 1].isalpha():
        j -= 1
    prefix = text[j:pos].lower()
    return prefix == "" or prefix.endswith(_LABEL_BEFORE_DATE)


def _has_embedded_date(text: str) -> bool:
    for m in _scan(_EMBED_DMY, text):
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        year_ok = 1995 <= int(y) <= 2035 if len(y) == 4 else 10 <= int(y) <= 30   # 2-digit: 2010-2030
        glued_to_digit = m.start() > 0 and text[m.start() - 1].isdigit()
        if glued_to_digit and len(m.group(1)) < 2:        # '13/13/13' must not match as '3/13/13'
            continue
        if year_ok and _label_ok(text, m.start()) and (
                (1 <= a <= 31 and 1 <= b <= 12) or (1 <= a <= 12 and 1 <= b <= 31)):  # dd/mm or mm/dd
            return True
    for m in _scan(_EMBED_YMD, text):
        if m.start() > 0 and text[m.start() - 1].isdigit():
            continue
        if 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31 and _label_ok(text, m.start()):
            return True
    if any(_label_ok(text, m.start()) for m in _scan(_EMBED_COMPACT, text)):
        return True
    return bool(_EMBED_NAME.search(text))


def find_date(words, rows) -> tuple[list[int], list[str]]:
    flags: list[str] = []
    cands = []  # (score, order, indices)
    order = 0
    for r in rows:
        row_has_kw = bool(re.search(r"\bdate|\bdt\b", _row_text(words, r), re.I))  # also matches "Date11/09/18"
        for i in r:
            if _has_embedded_date(words[i]["text"]):
                exact = _is_date(words[i]["text"])      # a clean '05-06-2018' beats '05-06-201820:23:14'
                cands.append((2 * row_has_kw + exact, order, [i]))
        for start in range(len(r)):
            for size in (1, 2, 3):
                idx = r[start:start + size]
                if len(idx) < size:
                    continue
                text = " ".join(words[i]["text"] for i in idx)
                if _is_date(text):
                    cands.append((2 * row_has_kw + 1, order, idx))
                    break  # prefer the shortest window at this start
        order += 1
    if not cands:
        return [], ["no_date"]
    best = sorted(cands, key=lambda c: (-c[0], c[1]))[0]
    distinct = {" ".join(words[i]["text"] for i in c[2]) for c in cands}
    if len(distinct) > 1:
        flags.append("multiple_dates")
    return best[2], flags


# ---------------------------------------------------------------- total
_TOTAL_KW = re.compile(
    r"jotal|tota[l1i]|amount\s*due|balance\s*due|amount\s*payable|net\s*amount|grand|to\s*pay|amt\s*due", re.I)
_EXCL = r"ex?c[il1]u?d|excl"                               # excluding / Exciuding / Ecluding (OCR slips)
_TOTAL_BAD = re.compile(
    r"su[bnm]\s*-?\s*total|total\s*(q|items?|discount|saving|point|tax|gst|vat|sst)|(tax|gst|vat|sst)\s*total"
    rf"|{_EXCL}|before|change|cash|tender|round(ing)?\s*adj|includes?\s*\d+\s*%",
    re.I,
)
_SMALL_RATIO = 0.12      # a tax column is at most ~6-10% of the amount column next to it


def _total_row_amount(amts: list[tuple[int, float]]) -> tuple[int, float]:
    """'TOTAL 73.87 4.43' -> the 73.87 (the last number is the tax column); else the rightmost amount."""
    if len(amts) >= 2 and amts[-1][1] <= _SMALL_RATIO * amts[-2][1]:
        return amts[-2]
    return amts[-1]


def find_total(words, rows) -> tuple[list[int], float | None, list[str]]:
    cands: list[tuple[int, float]] = []
    for ri, r in enumerate(rows):
        text = _row_text(words, r)
        if not _TOTAL_KW.search(text) or _TOTAL_BAD.search(text):
            continue
        amts = _amounts_in_row(words, r)
        if not amts and ri + 1 < len(rows):               # label on one row, value on the next
            nxt = rows[ri + 1]
            if not _TOTAL_KW.search(_row_text(words, nxt)):
                amts = _amounts_in_row(words, nxt)
        if amts:
            cands.append(_total_row_amount(amts))
    if not cands:
        return [], None, ["no_total"]
    flags = []
    values = {round(v, 2) for _, v in cands}
    if len(values) > 1:
        flags.append("multiple_totals")
    best = max(cands, key=lambda c: c[1])                # payable total >= subtotal
    return [best[0]], best[1], flags


# ---------------------------------------------------------------- tax
_TAX_KW = re.compile(r"\b(gst|vat|sst|cgst|sgst|igst|tax)\b", re.I)
_TAX_BAD = re.compile(
    r"tax\s*invoice|(tax|gst|vat|sst)\s*(id|reg|no|number|#)|reg\.?\s*no|inclusive|inc\.?\s*(gst|tax)"
    rf"|{_EXCL}|before|after|\bsummary\b|\bcode\b|\brate\b|\bsales\b|taxable|\bta\s*amt|\bamt\b|amount",
    re.I,
)
# "GST Summary  Amount(RM)  Tax(RM)" / "Amt (RM) Tax(RM)" - the header of the small table many receipts print
_TAX_HEADER = re.compile(r"(gst|tax|vat|sst)\W*su\w{1,3}a?ry|(amount|amaunt|amt)\W*(rm)?\W*tax", re.I)


def _table_tax(words, rows) -> tuple[int, float] | None:
    """First data row under a GST-summary header: '[code] [amount] [tax]' -> the rightmost number (tax)."""
    for ri, r in enumerate(rows):
        if not _TAX_HEADER.search(_row_text(words, r)):
            continue
        for nr in rows[ri + 1: ri + 4]:
            amts = [(i, v) for i in nr if (v := parse_amount(words[i]["text"], loose=True)) is not None]
            if len(amts) >= 2 and amts[-1][1] <= max(v for _, v in amts[:-1]):
                return amts[-1]
    return None


def find_tax(words, rows, total_value: float | None) -> tuple[list[int], list[str]]:
    cands: list[tuple[int, float]] = []
    tbl = _table_tax(words, rows)
    if tbl:
        cands.append(tbl)
    for r in rows:
        text = _row_text(words, r)
        amts = _amounts_in_row(words, r)
        if not amts:
            continue
        if _TAX_KW.search(text) and not _TAX_BAD.search(text):
            i, v = amts[-1]
        elif _TOTAL_KW.search(text) and not _TOTAL_BAD.search(text) and len(amts) >= 2 \
                and amts[-1][1] <= _SMALL_RATIO * amts[-2][1]:
            i, v = amts[-1]                              # 'TOTAL 73.87 4.43': the second number is the tax
        else:
            continue
        if total_value is None or v < total_value:
            cands.append((i, v))
    if not cands:
        return [], ["no_tax"]
    flags = ["multiple_taxes"] if len({round(v, 2) for _, v in cands}) > 1 else []
    return [cands[0][0]], flags


# ---------------------------------------------------------------- merchant
_NOT_MERCHANT = re.compile(
    r"receipt|invoice|\btax\b|gst|vat|\btel\b|phone|\bfax\b|\bdate\b|\btime\b|cashier|\bbill\b|\btable\b|www\.|http|@"
    r"|address|jalan|\broad\b|street|reg\b|thank|welcome|\bcopy\b|original|customer|order|slip|cash\s*sale|\bno\.?\s*\d",
    re.I,
)
_COMPANY_SUFFIX = re.compile(r"^(sdn|bhd|ltd|llp|llc|inc|corp|pvt|private|limited|trading|enterprises?|co\.?|company)\b", re.I)


def find_merchant(words, rows, height: int) -> tuple[list[int], list[str]]:
    limit = 0.25 * height
    skipped = 0
    for ri, r in enumerate(rows[:8]):
        y0 = min(words[i]["box"][1] for i in r)
        if y0 > limit and ri >= 3:
            break
        text = _row_text(words, r)
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        if letters < 3 or digits > 0.3 * len(text) or _NOT_MERCHANT.search(text):
            skipped += 1
            continue
        idx = list(r)
        if ri + 1 < len(rows):                           # "FRESH MART" / "SDN BHD" on two rows
            nxt = rows[ri + 1]
            if _COMPANY_SUFFIX.match(words[nxt[0]]["text"]):
                idx += nxt
        return idx, (["merchant_uncertain"] if skipped >= 2 else [])
    return [], ["no_merchant"]


# ---------------------------------------------------------------- public API
def auto_label(record: dict) -> tuple[list[str], list[str]]:
    """OCR record -> (entities per word, flags)."""
    words = record["words"]
    entities = ["O"] * len(words)
    if not words:
        return entities, ["no_words"]
    rows = group_rows(words)

    flags: list[str] = []
    total_idx, total_val, f = find_total(words, rows)
    flags += f
    date_idx, f = find_date(words, rows)
    flags += f
    tax_idx, f = find_tax(words, rows, total_val)
    flags += f
    merch_idx, f = find_merchant(words, rows, record["height"])
    flags += f

    # later assignments win on overlap; merchant/date last because they are position-based
    for name, idx in (("TAX", tax_idx), ("TOTAL", total_idx), ("DATE", date_idx), ("MERCHANT", merch_idx)):
        for i in idx:
            entities[i] = name
    return entities, flags
