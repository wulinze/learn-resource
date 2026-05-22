# TileKernels Puzzle

This directory contains standalone TileLang learning puzzles derived from the
current TileKernels implementations.

Each puzzle is split into:

- `starter.py`: the learner-owned implementation entrypoint.
- `answer.py`: a wrapper around the current production TileKernels kernel.
- `reference.py`: a PyTorch reference used by tests.
- `test_*.py`: standalone correctness tests for this puzzle.

Tests run the answer by default, so the repository remains green before a
learner fills in the starter:

```bash
pytest puzzles/levels/l03_reduction/stable_topk/test_stable_topk.py
```

To test a learner implementation:

```bash
TK_PUZZLE_IMPL=starter pytest puzzles/levels/l03_reduction/stable_topk/test_stable_topk.py
```

The project design and full learning roadmap live in
`docs/tilelang_puzzle_project.md`.

