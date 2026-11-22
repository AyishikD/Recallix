"""
Relative dates resolved at store time (config memory.resolve_dates).

"I went to the support group yesterday", stored on 8 May 2023, also means "7 May 2023". A memory keeps
its own text; the resolved dates are kept in metadata["dates"] and shown inline wherever the memory is read
(the answer check, and `content_resolved` in recall results):

    "...the support group yesterday [7 May 2023]..."
    "...joined a service-focused online group last week [the week before 1 January 2023]..."

Rules only, deterministic, no LLM. The reference date is the memory's timestamp (UTC). Only expressions with
a definite meaning are resolved; vague ones ("lately", "recently", "the other day", "a few weeks ago") are left
alone rather than guessed. Week-like expressions keep their relative form ("the week before <date>"), because a
week has no single date; weekday expressions also give the exact date ("last Friday" -> "9 September 2022, the
Friday before 14 September 2022").
"""
import re
from datetime import datetime, timedelta, timezone

from backend import settings

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
_NUM = r"(\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)"
_WD = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"


def _day(d):
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _month(d):
    return f"{d.strftime('%B')} {d.year}"


def _shift_month(d, n):
    m = d.month - 1 + n
    return d.replace(year=d.year + m // 12, month=m % 12 + 1, day=1)


def _n(word):
    return int(word) if word.isdigit() else NUMBERS[word.lower()]


def _last_weekday(ref, wd):   # "last friday": the most recent one strictly before the reference day
    back = (ref.weekday() - wd) % 7 or 7
    return ref - timedelta(days=back)


def _next_weekday(ref, wd):   # "next friday" / "this friday": the next one after the reference day
    ahead = (wd - ref.weekday()) % 7 or 7
    return ref + timedelta(days=ahead)


# (pattern, resolver(ref_date, match) -> text). Longer phrases first: the first match at a position wins.
_RULES = [
    (r"the day before yesterday", lambda r, m: _day(r - timedelta(days=2))),
    (r"the day after tomorrow", lambda r, m: _day(r + timedelta(days=2))),
    (r"yesterday(?: (?:morning|afternoon|evening|night))?|last night", lambda r, m: _day(r - timedelta(days=1))),
    (r"tomorrow(?: (?:morning|afternoon|evening|night))?", lambda r, m: _day(r + timedelta(days=1))),
    (r"today|tonight|this (?:morning|afternoon|evening)", lambda r, m: _day(r)),
    (rf"last {_WD}", lambda r, m: f"{_day(_last_weekday(r, WEEKDAYS.index(m.group(1).lower())))}, "
                                  f"the {m.group(1).capitalize()} before {_day(r)}"),
    (rf"(?:next|this|this coming) {_WD}", lambda r, m: f"{_day(_next_weekday(r, WEEKDAYS.index(m.group(1).lower())))}, "
                                                     f"the {m.group(1).capitalize()} after {_day(r)}"),
    (r"(?:last|this past|over the) weekend", lambda r, m: f"the weekend before {_day(r)}"),
    (r"this weekend", lambda r, m: f"the weekend of {_day(r)}"),
    (r"next weekend", lambda r, m: f"the weekend after {_day(r)}"),
    (r"last week|this past week|the past week", lambda r, m: f"the week before {_day(r)}"),
    (r"this week|earlier this week", lambda r, m: f"the week of {_day(r)}"),
    (r"next week", lambda r, m: f"the week after {_day(r)}"),
    (r"last month", lambda r, m: _month(_shift_month(r, -1))),
    (r"this month|earlier this month", lambda r, m: _month(r)),
    (r"next month", lambda r, m: _month(_shift_month(r, 1))),
    (r"last year", lambda r, m: str(r.year - 1)),
    (r"this year|earlier this year", lambda r, m: str(r.year)),
    (r"next year", lambda r, m: str(r.year + 1)),
    (rf"{_NUM} days? ago", lambda r, m: _day(r - timedelta(days=_n(m.group(1))))),
    (rf"{_NUM} weeks? ago", lambda r, m: f"the week of {_day(r - timedelta(weeks=_n(m.group(1))))}"),
    (rf"{_NUM} months? ago", lambda r, m: _month(_shift_month(r, -_n(m.group(1))))),
    (rf"{_NUM} years? ago", lambda r, m: str(r.year - _n(m.group(1)))),
    (rf"in {_NUM} days?", lambda r, m: _day(r + timedelta(days=_n(m.group(1))))),
    (rf"in {_NUM} weeks?", lambda r, m: f"the week of {_day(r + timedelta(weeks=_n(m.group(1))))}"),
]
_COMPILED = [(re.compile(rf"\b(?:{p})\b", re.I), fn) for p, fn in _RULES]


def enabled():
    return bool((settings.get("memory.resolve_dates", {}) or {}).get("enabled", True))


def resolve(text, timestamp):
    """[{"text", "start", "end", "value"}] for each definite relative date in `text`, said at `timestamp`."""
    if not text or timestamp is None:
        return []
    ref = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).date()
    found = []
    taken = []
    for rx, fn in _COMPILED:
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue  # inside a longer phrase already resolved
            try:
                value = fn(ref, m)
            except (KeyError, ValueError, OverflowError):
                continue
            taken.append((m.start(), m.end()))
            found.append({"text": m.group(0), "start": m.start(), "end": m.end(), "value": value})
    return sorted(found, key=lambda d: d["start"])


def annotate(text, dates):
    """The text with each resolved date added in brackets after its expression."""
    if not dates:
        return text
    out, last = [], 0
    for d in sorted(dates, key=lambda d: d["start"]):
        s, e = int(d["start"]), int(d["end"])
        if s < last or text[s:e].lower() != str(d["text"]).lower():
            continue  # spans no longer match the text: skip rather than misplace
        out.append(text[last:e]); out.append(f" [{d['value']}]"); last = e
    out.append(text[last:])
    return "".join(out)
