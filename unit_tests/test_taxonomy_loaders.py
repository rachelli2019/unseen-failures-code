"""Unit tests for the fail-loudly validation in the taxonomy loaders:
av_discovery_fixed_primary.load_session_to_failure_mode and
av_discovery_fixed_set_valued.load_session_to_failure_modes, on small
malformed JSON fixtures written to temp files. Both loaders read this
project's native clustering/LLM-merge output shape directly (a top-level
`modes` list) -- no separate reshaping step.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from av_discovery_fixed_primary import load_session_to_failure_mode
from av_discovery_fixed_set_valued import load_session_to_failure_modes


def _write_taxonomy(data: dict) -> Path:
    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp_file)
    tmp_file.close()
    return Path(tmp_file.name)


class TestLoadSessionModeLookupErrors(unittest.TestCase):
    def test_missing_modes_key_raises(self) -> None:
        path = _write_taxonomy({})
        with self.assertRaises(ValueError):
            load_session_to_failure_mode(path)

    def test_empty_modes_list_raises(self) -> None:
        path = _write_taxonomy({"modes": []})
        with self.assertRaises(ValueError):
            load_session_to_failure_mode(path)

    def test_mode_with_empty_member_session_ids_raises(self) -> None:
        path = _write_taxonomy({"modes": [{"mode_id": "m0", "member_session_ids": []}]})
        with self.assertRaises(ValueError):
            load_session_to_failure_mode(path)

    def test_duplicate_session_id_within_one_mode_raises(self) -> None:
        path = _write_taxonomy({"modes": [{"mode_id": "m0", "member_session_ids": ["s0", "s0"]}]})
        with self.assertRaises(ValueError):
            load_session_to_failure_mode(path)

    def test_session_under_two_modes_raises(self) -> None:
        path = _write_taxonomy(
            {
                "modes": [
                    {"mode_id": "m0", "member_session_ids": ["s0"]},
                    {"mode_id": "m1", "member_session_ids": ["s0"]},
                ]
            }
        )
        with self.assertRaises(ValueError):
            load_session_to_failure_mode(path)

    def test_well_formed_taxonomy_loads_cleanly(self) -> None:
        path = _write_taxonomy(
            {
                "modes": [
                    {"mode_id": "m0", "member_session_ids": ["s0"]},
                    {"mode_id": "m1", "member_session_ids": ["s1", "s2"]},
                ]
            }
        )
        lookup = load_session_to_failure_mode(path)
        self.assertEqual(lookup, {"s0": "m0", "s1": "m1", "s2": "m1"})


class TestLoadSessionToFailureModesErrors(unittest.TestCase):
    def test_missing_modes_key_raises(self) -> None:
        path = _write_taxonomy({})
        with self.assertRaises(ValueError):
            load_session_to_failure_modes(path)

    def test_empty_modes_list_raises(self) -> None:
        path = _write_taxonomy({"modes": []})
        with self.assertRaises(ValueError):
            load_session_to_failure_modes(path)

    def test_mode_with_empty_member_raw_ids_raises(self) -> None:
        path = _write_taxonomy({"modes": [{"mode_id": "m0", "member_raw_ids": []}]})
        with self.assertRaises(ValueError):
            load_session_to_failure_modes(path)

    def test_malformed_raw_id_without_separator_raises(self) -> None:
        path = _write_taxonomy({"modes": [{"mode_id": "m0", "member_raw_ids": ["no_separator"]}]})
        with self.assertRaises(ValueError):
            load_session_to_failure_modes(path)

    def test_duplicate_raw_id_across_modes_raises(self) -> None:
        # Unlike the primary loader, a session under multiple modes is fine
        # here (that's the whole point of set-valued); what's invalid is the
        # exact same raw ASSIGNMENT id appearing under two different modes.
        path = _write_taxonomy(
            {
                "modes": [
                    {"mode_id": "m0", "member_raw_ids": ["s0::a"]},
                    {"mode_id": "m1", "member_raw_ids": ["s0::a"]},
                ]
            }
        )
        with self.assertRaises(ValueError):
            load_session_to_failure_modes(path)

    def test_session_under_multiple_modes_is_allowed(self) -> None:
        path = _write_taxonomy(
            {
                "modes": [
                    {"mode_id": "m0", "member_raw_ids": ["s0::a"]},
                    {"mode_id": "m1", "member_raw_ids": ["s0::b"]},
                ]
            }
        )
        session_to_modes = load_session_to_failure_modes(path)
        self.assertEqual(session_to_modes["s0"], frozenset({"m0", "m1"}))


if __name__ == "__main__":
    unittest.main()
