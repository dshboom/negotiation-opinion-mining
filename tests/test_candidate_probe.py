"""CPU-only synthetic checks. No model or competition data needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from candidate_probe import atomic, coverage, load


class ExactBGE:
    def cos(self, a, b):
        return float(a == b)


def card(name, stance="support"):
    return {"issue_name": name, "stance": stance, "chain": [name]}


class ProbeTests(unittest.TestCase):
    def test_exact_and_stance(self):
        self.assertEqual(coverage([card("a")], [card("a")], ExactBGE(), .7)[0], 1)
        self.assertEqual(coverage([card("a", "oppose")], [card("a")], ExactBGE(), .7)[0], 0)

    def test_duplicate_cannot_cover_two(self):
        self.assertEqual(coverage([card("a")], [card("a"), card("a")], ExactBGE(), .7)[0], 1)

    def test_empty(self):
        self.assertEqual(coverage([], [card("a")], ExactBGE(), .7), (0, []))

    def test_strict_threshold(self):
        self.assertEqual(coverage([card("a")], [card("a")], ExactBGE(), 1)[0], 0)

    def test_cardinality_before_similarity(self):
        class MatrixBGE:
            def cos(self, a, b):
                return {("p1", "g1"): 1.0, ("p1", "g2"): .71,
                        ("p2", "g1"): .71, ("p2", "g2"): .69}[a, b]
        self.assertEqual(coverage([card("p1"), card("p2")],
                                  [card("g1"), card("g2")], MatrixBGE(), .7)[0], 2)

    def test_io(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            atomic(p, {"sample_id": "a"})
            self.assertEqual(json.loads(p.read_text())["sample_id"], "a")
            p.write_text(json.dumps({"sample_id": "a"}))
            self.assertEqual(load(p)[0]["sample_id"], "a")
            p.write_text('\n'.join([json.dumps({"sample_id": "a"})] * 2))
            with self.assertRaises(ValueError):
                load(p)


if __name__ == "__main__":
    unittest.main()
