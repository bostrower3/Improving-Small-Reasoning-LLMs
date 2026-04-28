import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class SumSample:
    query: str
    response: str
    num_vars: int
    ask_i: int
    ask_j: int
    ans: int
    values: list[int]

#note that min_vars is inclusive, max_vars is EXCLUSIVE, so num vars is [min_vars, max_vars)
def generate_sample(rng: random.Random, min_vars: int, max_vars: int) -> SumSample:
    numVars = rng.randint(min_vars, max_vars-1)
    values = [rng.randint(10, 29) for _ in range(numVars)]
    ask_i, ask_j = rng.sample(range(numVars), 2)
    answer = values[ask_i] + values[ask_j]
    varStr = ", ".join(f"Var{i}={values[i]}" for i in range(numVars))
    queryStr = f"<BOS> {varStr}, Var{ask_i}+Var{ask_j}="
    responseStr = f"{values[ask_i]}+{values[ask_j]}={answer}<EOS>"

    return SumSample(
        query=queryStr, 
        response=responseStr, 
        num_vars = numVars, 
        ask_i = ask_i, 
        ask_j = ask_j, 
        ans=answer, 
        values=values)

def make_split(size, min_vars, max_vars, seed:int) -> list[SumSample]:
    rng = random.Random(seed)
    return [generate_sample(rng, min_vars, max_vars) for _ in range(size)]


def write_jsonl(path: str | Path, samples: list[SumSample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # This file is in dataGeneration/; repo root is one level up.
    data_dir = Path(__file__).resolve().parent.parent / "data" / "summation"

    write_jsonl(data_dir / "train.jsonl", make_split(4096, 5, 15, seed=1))
    write_jsonl(data_dir / "val.jsonl", make_split(1024, 5, 15, seed=2))

    test_ranges = [(5, 15), (15, 30), (30, 50), (50, 70), (70, 90)]
    for idx, (lo, hi) in enumerate(test_ranges, start=1):
        write_jsonl(
            data_dir / f"test_{lo}_{hi}.jsonl",
            make_split(1024, lo, hi, seed=100 + idx),
        )