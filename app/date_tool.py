
import json
import re
import sys
from datetime import date, datetime


# ---------- Configuration ----------

OPERATIONS = ("today", "days_between")

FORMATS = (
    "%Y-%m-%d",      # 2026-08-02
    "%d %B %Y",      # 2 August 2026
    "%d %b %Y",      # 2 Aug 2026
    "%B %d %Y",      # August 2 2026 (comma removed below)
    "%b %d %Y",      # Aug 2 2026
)


# ---------- Helpers ----------

def parse_date(value: str) -> date | None:
    """Read a date written as 2026-08-02, '2 August 2026' or 'August 2, 2026'.

    Day/month numeric formats like 02/08/2026 are refused on purpose,
    because they are ambiguous (2 August or February 8?).
    """
    if not isinstance(value, str):
        return None

    text = " ".join(value.replace(",", " ").split())
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)   # 2nd -> 2

    for pattern in FORMATS:
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue

    return None


def describe(days: int) -> str:
    """Human-friendly size of a day count."""
    size = abs(days)
    return f"about {size / 7:.1f} weeks, {size / 30.44:.1f} months, {size / 365.25:.2f} years"


# ---------- date_tool ----------

def date_tool(operation: str, start: str | None = None,
              end: str | None = None) -> dict:
    """Return today's date, or the days from start to end."""
    if operation not in OPERATIONS:
        return {"ok": False,
                "error": f"Operation must be one of: {', '.join(OPERATIONS)}."}

    if operation == "today":
        today = date.today()
        return {
            "ok": True,
            "operation": "today",
            "today": today.isoformat(),
            "weekday": today.strftime("%A"),
            "readable": today.strftime("%d %B %Y").lstrip("0"),
        }

    # days_between: 'today' is accepted as either date.
    dates = {}
    for name, value in (("start", start), ("end", end)):
        if isinstance(value, str) and value.strip().lower() == "today":
            dates[name] = date.today()
        else:
            dates[name] = parse_date(value)

        if dates[name] is None:
            return {
                "ok": False,
                "error": (f"Could not read the {name} date {value!r}. Use "
                          "YYYY-MM-DD, '2 August 2026' or 'today'."),
            }

    days = (dates["end"] - dates["start"]).days
    today = date.today()

    # A plain sentence, so the model cannot mix up "ago" and "from now".
    def words(day: date) -> str:
        return f"{day.day} {day:%B %Y}"

    start, end = dates["start"], dates["end"]
    if end == today and start != today:
        plain = (f"{words(start)} was {days} days ago (in the past)" if days > 0
                 else f"{words(start)} is {-days} days from now (in the future)")
    elif start == today and end != today:
        plain = (f"{words(end)} is {days} days from now (in the future)" if days > 0
                 else f"{words(end)} was {-days} days ago (in the past)")
    else:
        plain = (f"{words(end)} is {abs(days)} days "
                 f"{'after' if days >= 0 else 'before'} {words(start)}")

    return {
        "ok": True,
        "operation": "days_between",
        "start": dates["start"].isoformat(),
        "end": dates["end"].isoformat(),
        "days": days,
        "direction": ("end is after start" if days > 0 else
                      "end is before start" if days < 0 else "same day"),
        "approximately": describe(days),
        "plain": plain,
    }


# ---------- Tool schema for the agents ----------

DATE_TOOL = {
    "type": "function",
    "function": {
        "name": "date_tool",
        "description": (
            "Date calculations. operation='today' returns today's date. "
            "operation='days_between' returns the number of days from start "
            "to end (negative if end is earlier). Dates: YYYY-MM-DD, "
            "'2 August 2026', or 'today'. Use it for any question about how "
            "long until / since a date; never count days yourself. Only use "
            "dates you have read in a tool result (e.g. from get_article), "
            "never dates from memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": list(OPERATIONS)},
                "start": {"type": "string",
                          "description": "Start date (for days_between)."},
                "end": {"type": "string",
                        "description": "End date (for days_between)."},
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
    },
}


# ---------- Run from the terminal ----------

def main() -> None:
    """Usage: python app/date_tool.py today | days_between START END"""
    if len(sys.argv) < 2:
        print("Usage: python app/date_tool.py today\n"
              "       python app/date_tool.py days_between 2024-08-01 \"2 August 2026\"")
        return

    operation = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else None
    end = sys.argv[3] if len(sys.argv) > 3 else None
    print(json.dumps(date_tool(operation, start, end), indent=2))


if __name__ == "__main__":
    main()
