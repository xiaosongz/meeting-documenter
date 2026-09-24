import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Fictional registry: tests never read the user's private references/KNOWN_SPEAKERS.yaml.
REGISTRY = """\
speakers:
  - canonical_name: Alice Example
    aliases: [Alice, Alicia]
  - canonical_name: Bob Example
    aliases: [Bob, 鲍勃]
  - canonical_name: Carol Example
    aliases: [Carol]
  - canonical_name: Dave Example
    aliases: [Dave]
"""


@pytest.fixture(autouse=True)
def synthetic_environment(tmp_path_factory, monkeypatch):
    reg = tmp_path_factory.getbasetemp() / "KNOWN_SPEAKERS.yaml"
    if not reg.exists():
        reg.write_text(REGISTRY, encoding="utf-8")
    monkeypatch.setenv("MEETING_DOCUMENTER_KNOWN_SPEAKERS", str(reg))
    for var in ("VOICEPRINT_DIR", "VOICEPRINT_FORBIDDEN_ROOTS", "VAULT_PATH", "MEETING_NOTES_DIR"):
        monkeypatch.delenv(var, raising=False)
