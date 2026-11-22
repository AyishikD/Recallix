"""Relative dates resolved at store time (backend/brain/features/dates.py)."""
import calendar
from datetime import datetime

from backend.brain.features.dates import annotate, resolve


def _ts(s):
    return calendar.timegm(datetime.strptime(s, "%d %B %Y").timetuple())


def _a(text, when):
    return annotate(text, resolve(text, _ts(when)))


def test_relative_dates_become_real_dates_in_locomo_style():
    assert _a("got mentored yesterday!", "16 June 2023") == "got mentored yesterday [15 June 2023]!"
    assert _a("I'll come over tomorrow", "4 November 2022") == "I'll come over tomorrow [5 November 2022]"
    assert _a("watched it last night", "2 May 2022") == "watched it last night [1 May 2022]"
    assert _a("joined a group last week", "1 January 2023") == "joined a group last week [the week before 1 January 2023]"
    assert _a("the roadtrip this past weekend", "20 October 2023") == "the roadtrip this past weekend [the weekend before 20 October 2023]"
    assert _a("Last Friday, I made a dessert", "14 September 2022") == \
        "Last Friday [9 September 2022, the Friday before 14 September 2022], I made a dessert"
    assert _a("moving next month", "31 January 2023") == "moving next month [February 2023]"
    assert _a("my 18th birthday ten years ago", "27 June 2023") == "my 18th birthday ten years ago [2013]"


def test_last_weekday_on_that_weekday_means_a_week_earlier():
    assert "9 September 2022" in _a("last Friday", "16 September 2022")   # 16 Sep 2022 was a Friday


def test_vague_words_are_left_alone():
    text = "Lately I've been busy, recently too, the other day, and a few weeks ago"
    assert _a(text, "2 May 2022") == text


def test_recall_shows_resolved_dates_to_the_reader_and_the_caller():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        with box.silenced():
            box.mm.store("u", "I went to the support group yesterday, it was great", timestamp=_ts("8 May 2023"))
            out = box.mm.retrieve("u", "When did I go to the support group?")
        top = out["memories"][0]
        assert top["content"] == "I went to the support group yesterday, it was great"   # own text unchanged
        assert top["content_resolved"] == "I went to the support group yesterday [7 May 2023], it was great"
        assert top["metadata"]["dates"][0]["value"] == "7 May 2023"
