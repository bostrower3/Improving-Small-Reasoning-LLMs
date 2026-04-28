import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GenerationSample:
    query: str
    response: str
    query_equations: int
    response_equations: int
    max_new_tokens: int | None
    start_left: str
    start_right: str


def apply_op(a1: int, a2: int, b1: int, b2: int) -> tuple[int, int]:
    """
    f1(a1, b1) = |a1 + b1| mod 9
    f2(a2, b2) = |a2 - b2| mod 9
    """
    f1 = abs(a1 + b1) % 9
    f2 = abs(a2 - b2) % 9
    return f1, f2


def step_equation(left: tuple[int, int], right: tuple[int, int]) -> tuple[str, tuple[int, int], tuple[int, int]]:
    a1, a2 = left
    b1, b2 = right
    out = apply_op(a1, a2, b1, b2)
    equation = f"{a1}{a2}@{b1}{b2}={out[0]}{out[1]},"
    # Next equation uses old right as new left and output as new right.
    return equation, right, out


def build_equation_chain(
    rng: random.Random,
    query_equations: int,
    response_equations: int,
) -> GenerationSample:
    # Initial digits are sampled uniformly from 0-9.
    left = (rng.randint(0, 9), rng.randint(0, 9))
    right = (rng.randint(0, 9), rng.randint(0, 9))

    start_left = f"{left[0]}{left[1]}"
    start_right = f"{right[0]}{right[1]}"

    query_parts = ["<BOS> "]
    for _ in range(query_equations):
        eq, left, right = step_equation(left, right)
        query_parts.append(eq)
    query = "".join(query_parts)

    response_parts: list[str] = []
    for _ in range(response_equations):
        eq, left, right = step_equation(left, right)
        response_parts.append(eq)
    response = "".join(response_parts)

    return GenerationSample(
        query=query,
        response=response,
        query_equations=query_equations,
        response_equations=response_equations,
        max_new_tokens=None,
        start_left=start_left,
        start_right=start_right,
    )


def make_train_or_val_split(
    size: int,
    query_equations: int,
    min_response_equations: int,
    max_response_equations: int,
    seed: int,
) -> list[GenerationSample]:
    rng = random.Random(seed)
    out: list[GenerationSample] = []
    for _ in range(size):
        response_equations = rng.randint(min_response_equations, max_response_equations)
        sample = build_equation_chain(
            rng=rng,
            query_equations=query_equations,
            response_equations=response_equations,
        )
        out.append(sample)
    return out


def make_test_split(
    size: int,
    query_equations: int,
    max_new_tokens: int,
    seed: int,
) -> list[GenerationSample]:
    # Each equation uses 6 tokens under the target tokenizer.
    response_equations = max_new_tokens // 6
    rng = random.Random(seed)
    out: list[GenerationSample] = []
    for _ in range(size):
        sample = build_equation_chain(
            rng=rng,
            query_equations=query_equations,
            response_equations=response_equations,
        )
        sample.max_new_tokens = max_new_tokens
        out.append(sample)
    return out


def write_jsonl(path: str | Path, samples: list[GenerationSample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # This file is in dataGeneration/; repo root is one level up.
    data_dir = Path(__file__).resolve().parent.parent / "data" / "generation"

    write_jsonl(
        data_dir / "train.jsonl",
        make_train_or_val_split(
            size=4096,
            query_equations=5,
            min_response_equations=1,
            max_response_equations=10,
            seed=11,
        ),
    )
    write_jsonl(
        data_dir / "val.jsonl",
        make_train_or_val_split(
            size=1024,
            query_equations=5,
            min_response_equations=1,
            max_response_equations=10,
            seed=12,
        ),
    )

    test_setups = [(10, 60), (30, 180), (50, 300), (60, 360)]
    for idx, (num_equations, max_new_tokens) in enumerate(test_setups, start=1):
        expected = max_new_tokens // 6
        if expected != num_equations:
            raise ValueError(
                f"Token/equation mismatch: expected {expected} equations for {max_new_tokens} tokens, got {num_equations}"
            )

        write_jsonl(
            data_dir / f"test_{num_equations}_{max_new_tokens}.jsonl",
            make_test_split(
                size=1024,
                query_equations=5,
                max_new_tokens=max_new_tokens,
                seed=200 + idx,
            ),
        )
