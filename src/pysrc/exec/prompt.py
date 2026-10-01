import json
from anthropic import Anthropic

_client = Anthropic()

_SYSTEM = """
You map a natural language melody description to a structured set of musical parameters.
Return ONLY a valid JSON object with exactly these keys and valid values:

{
  "bpm":     integer, one of: 60,70,80,90,100,110,120,130,140,150,160,170,180
  "ts":      one of: "4/4", "3/4", "6/8", "9/8", "12/8"
  "bars":    integer 2-80
  "first":   integer 21-108 (MIDI note number — middle C is 60)
  "last":    integer 21-108
  "mode":    one of: "major", "minor"
  "genre":   one of: "Pop", "Rock", "Funk/Soul", "R&B", "Hip-hop", "Other"
  "era":     one of: "1950s","1960s","1970s","1980s","1990s","2000s","2010s","2020s"
  "contour": one of: "ascending", "descending", "arch", "valley"
  "density": one of: "sparse", "moderate", "dense"
  "range":   one of: "narrow", "moderate", "wide"
}

Guidelines:
- mode: minor for sad/dark/tense, major for happy/bright/uplifting
- contour: ascending = rises overall, descending = falls, arch = rises then falls, valley = falls then rises
- density: sparse = few notes per bar, dense = many notes
- range: narrow < 1 octave, moderate = 1-2 octaves, wide > 2 octaves
- first/last: 60 = middle C. Use ~60-72 for mid-range melodies. Set last higher than first for ascending contour, lower for descending.
- If the description is ambiguous, choose musically sensible defaults.
""".strip()

_VALID = {
    "bpm":     {60,70,80,90,100,110,120,130,140,150,160,170,180},
    "ts":      {"4/4","3/4","6/8","9/8","12/8"},
    "mode":    {"major","minor"},
    "genre":   {"Pop","Rock","Funk/Soul","R&B","Hip-hop","Other"},
    "era":     {"1950s","1960s","1970s","1980s","1990s","2000s","2010s","2020s"},
    "contour": {"ascending","descending","arch","valley"},
    "density": {"sparse","moderate","dense"},
    "range":   {"narrow","moderate","wide"},
}

_DEFAULTS = {
    "bpm": 120, "ts": "4/4", "bars": 16, "first": 60, "last": 60,
    "mode": "major", "genre": "Pop", "era": "2000s",
    "contour": "arch", "density": "moderate", "range": "moderate",
}


def _clamp(val: int, lo: int, hi: int, step: int = 1) -> int:
    val = max(lo, min(hi, val))
    if step > 1:
        val = round(val / step) * step
    return val


def parse_prompt(text: str) -> dict:
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=_SYSTEM,
        messages=[{"role": "user", "content": text}]
    )

    try:
        params = json.loads(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        return _DEFAULTS.copy()

    result = {}
    result["bpm"]     = _clamp(int(params.get("bpm", 120)), 60, 180, 10)
    result["ts"]      = params.get("ts", "4/4") if params.get("ts") in _VALID["ts"] else "4/4"
    result["bars"]    = _clamp(int(params.get("bars", 16)), 2, 80)
    result["first"]   = _clamp(int(params.get("first", 60)), 21, 108)
    result["last"]    = _clamp(int(params.get("last", 60)), 21, 108)

    for key in ("mode", "genre", "era", "contour", "density", "range"):
        val = params.get(key, _DEFAULTS[key])
        result[key] = val if val in _VALID[key] else _DEFAULTS[key]

    return result