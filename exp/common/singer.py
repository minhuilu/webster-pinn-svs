# file: exp/common/singer.py
import os, pathlib, sys
# Path fix, consistent with the other project scripts
if __name__ == "__main__" and not __package__:
    current_file = pathlib.Path(__file__).resolve()
    repo_root = current_file.parent.parent
    sys.path.insert(0, str(repo_root))
    __package__ = "exp.common"

# Shared speaker presets; female is the default
SPEAKER_PROFILES = {
    "female": {
        "L": 0.155,                         # shorter female/neutral vocal tract, yielding higher resonances
        "f0_by_vowel": {"a": 200.0, "i": 240.0, "u": 180.0},
        "vibrato": {"depth": 6.0, "rate": 6.0},
    },
    "neutral": {
        "L": 0.165,
        "f0_by_vowel": {"a": 180.0, "i": 220.0, "u": 160.0},
        "vibrato": {"depth": 5.0, "rate": 5.5},
    },
    # male is kept as an option but is not the default
    "male": {
        "L": 0.175,
        "f0_by_vowel": {"a": 140.0, "i": 200.0, "u": 120.0},
        "vibrato": {"depth": 4.5, "rate": 5.0},
    },
}

DEFAULT_SPEAKER = os.environ.get("SPEAKER_PROFILE", "female").lower()

def get_profile(name: str = None):
    name = (name or DEFAULT_SPEAKER).lower()
    if name not in SPEAKER_PROFILES:
        name = "female"
    return SPEAKER_PROFILES[name]

def band_scale_for_L(L: float) -> float:
    """
    Scale formant search bands by vocal-tract length: scale ~= 0.17 / L
    0.17 is a common nominal neutral length; shorter L shifts bands upward.
    """
    return float(0.17) / float(L)
