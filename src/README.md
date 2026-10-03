# Day 17 implementation

The Python files in this folder implement the offline memory lab. The offline path is deterministic and needs no API key. `model_provider.py` also provides constructors for the six optional live providers listed in the root README.

Run from the repository root:

```bash
python3 src/benchmark.py
python3 -m pytest src/test_agents.py -v
```

The benchmark prints Standard and Long-Context Stress tables. It uses temporary isolated state so repeated runs are comparable. For measurements and limitations, see `Analysis.md` in the repository root.
