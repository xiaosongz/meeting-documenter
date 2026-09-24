"""PII guard (scripts/dev/check_pii.py). Sensitive-looking strings are assembled at runtime so
this file itself stays clean under the scan."""
import importlib.util
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_pii", ROOT / "scripts" / "dev" / "check_pii.py")
pii = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pii)

HOME_PATH = "/" + "Users" + "/someone/notes"
VOLUME_PATH = "/" + "Volumes" + "/Disk/archive"
LINUX_HOME = "/" + "home" + "/someone/"
PRIVATE_EMAIL = "someone" + "@" + "private-mail.org"
REGISTRY = """speakers:
  - canonical_name: Quentin Placeholderson
    aliases: [Quentin, "Q. Placeholderson", 张三丰, Bob, QP]
    email: quentin""" + "@" + """corp-internal.net
    notes: "also reachable at qp""" + "@" + """other-internal.org"
"""


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = pii.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class CheckPiiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.roster = self.dir / "KNOWN_SPEAKERS.yaml"
        self.roster.write_text(REGISTRY, encoding="utf-8")

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_generic_rules(self):
        bad = self.write("bad.md", f"see {HOME_PATH}\n{VOLUME_PATH}\n{LINUX_HOME}\nmail {PRIVATE_EMAIL}\n")
        rc, out, _ = run("--no-roster", bad)
        self.assertEqual(rc, 1)
        self.assertEqual(out.count(": path: "), 3)
        self.assertEqual(out.count(": email: "), 1)
        self.assertNotIn("someone", out)                                   # findings are masked
        ok = self.write("ok.md", "name@example.com, a@b.example, x@y.invalid, ~/notes, /path/to/vault\n")
        self.assertEqual(run("--no-roster", ok)[0], 0)

    def test_forbidden_files(self):
        for name in ("voiceprints.sqlite", "calibration.json", "x Meeting-Transcript.utterances.json",
                     "clip.wav", "talk.m4a", "emb.npy", ".env", "references/KNOWN_SPEAKERS.yaml"):
            with self.subTest(name=name):
                self.assertTrue(pii.check_name(name))
        for name in (".env.example", "references/KNOWN_SPEAKERS.template.yaml", "tests/test_pipeline.py"):
            with self.subTest(name=name):
                self.assertEqual(pii.check_name(name), [])

    def test_roster_denylist(self):
        text = ("Quentin said hi\nq. placeholderson agreed\n张三丰 joined\nquentinx is fine\n"
                "Bob is a placeholder\nQP is too short\n")
        path = self.write("notes.md", text)
        rc, out, err = run("--roster", str(self.roster), path)
        self.assertEqual(rc, 1)
        self.assertEqual(out.count(": roster: "), 3, out)                   # Quentin, Q. Placeholderson, 张三丰
        self.assertIn(":1: roster", out)
        self.assertNotIn(":4: roster", out)                                 # whole-word only
        self.assertNotIn(":5: roster", out)                                 # bare placeholder name
        self.assertIn("skipped", err)
        emails = self.write("mail.md", "quentin" + "@" + "corp-internal.net\n")
        self.assertEqual(run("--roster", str(self.roster), emails)[0], 1)
        self.assertEqual(run("--no-roster", self.write("clean.md", "Quentin\n"))[0], 0)

    def test_license_is_exempt_from_roster_only(self):
        self.assertFalse(pii.roster_applies("LICENSE"))
        self.assertTrue(pii.roster_applies("README.md"))

    def test_patch_scan_checks_messages_added_lines_and_names(self):
        patch = ("\x1ecommit 0123456789abcdef\nfeat: thanks Quentin\n\n"
                 "diff --git a/docs/a.md b/docs/a.md\n--- a/docs/a.md\n+++ b/docs/a.md\n@@ -1 +1,2 @@\n"
                 f"-old {HOME_PATH}\n+new line\n+{PRIVATE_EMAIL}\n"
                 "diff --git a/store/voiceprints.sqlite b/store/voiceprints.sqlite\nBinary files differ\n")
        roster, _ = pii.roster_patterns(pii.load_roster_terms(self.roster), 3)
        found = pii.scan_patch(patch, roster, "log")
        kinds = sorted(f.split(": ")[1] for f in found)
        self.assertEqual(kinds, ["email", "file", "roster"], found)        # removed line ignored
        self.assertTrue(any(f.startswith("docs/a.md:2: email") for f in found), found)
        self.assertTrue(any(f.startswith("log message 0123456789ab:1: roster") for f in found), found)

    def test_repository_is_clean_without_roster(self):
        result = subprocess.run(["python3", str(ROOT / "scripts" / "dev" / "check_pii.py"), "--no-roster"],
                                capture_output=True, text=True, timeout=60, cwd=ROOT)
        if "git failed" in result.stderr:
            self.skipTest("not a git checkout")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
