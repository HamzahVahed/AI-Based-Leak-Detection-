# naming.py
import re

# Words/tokens to strip from stems when normalizing
SENSOR_TOKENS = {
    "accelerometer", "accel", "acc", "a",
    "hydrophone", "hydrophones", "hyd", "h",
    "pressure", "dynamicpressure", "dynpress", "dps", "p",
    "sensor", "looped", "cwt", "scalogram", "spectrogram", "spec",
}

CLASS_TOKENS = {
    "no", "leak", "noleak", "orifice", "gasket", "longitudinal", "circumferential", "crack",
}

# Any extra fluff to remove everywhere
EXTRA_TOKENS = {"img", "image", "sample", "seg", "slice", "frame"}

TOKEN_SET = SENSOR_TOKENS | CLASS_TOKENS | EXTRA_TOKENS

_non_alnum = re.compile(r"[^0-9a-z]+")
_spaces = re.compile(r"\s+")

def _tokenize(s: str):
    s = s.lower()
    s = _non_alnum.sub(" ", s)     # keep only [0-9a-z] and turn the rest into spaces
    s = _spaces.sub(" ", s).strip()
    return [t for t in s.split(" ") if t]

def _digits_id(s: str):
    # Return the **longest** digit run (e.g., 000123 or 123), or None
    runs = re.findall(r"\d+", s)
    if not runs:
        return None
    # Prefer the longest; tie-breaker = last occurrence
    runs.sort(key=lambda x: (len(x), x))
    return runs[-1]

def normalize_stem(stem: str) -> str:
    """
    Create a robust, comparable key across sensors.
    Strategy:
      1) If a numeric ID exists, return that.
      2) Otherwise, drop known tokens (sensor names, class words, fluff),
         and return the remaining alphanumeric tokens concatenated.
    """
    # Fast path: numeric id
    num = _digits_id(stem)
    if num:
        return num.lstrip("0") or "0"  # "0003" -> "3", keep "0" if all zeros

    # Token path
    toks = _tokenize(stem)
    toks = [t for t in toks if t not in TOKEN_SET]
    if not toks:
        # Fall back to compacted alnum of the original stem
        compact = re.sub(r"[^0-9a-z]+", "", stem.lower())
        return compact or stem.lower()
    return "".join(toks)
