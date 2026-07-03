"""Score predictions.json on exact match and execution validity.

Metrics:
  exact match         normalized string equality with the gold SQL (a lower
                      bound — a correct query written differently still counts
                      as a miss)
  execution validity  the query runs without error against the schema built in
                      an in-memory SQLite database (catches hallucinated tables,
                      wrong columns, and syntax errors; tables are empty, so
                      result equality is not checked)
"""
import argparse
import json
import re
import sqlite3


def normalize_sql(sql):
    sql = sql.strip().rstrip(";").lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql.replace('"', "'")


def exact_match(pred, gold):
    return normalize_sql(pred) == normalize_sql(gold)


def executes_ok(context, sql):
    conn = sqlite3.connect(":memory:")
    try:
        for stmt in context.split(";"):
            if stmt.strip():
                conn.execute(stmt)
        conn.execute(sql)
        return True
    except Exception:
        return False
    finally:
        conn.close()


def score(records, key):
    em = sum(exact_match(r[key], r["gold"]) for r in records) / len(records)
    ev = sum(executes_ok(r["context"], r[key]) for r in records) / len(records)
    return em, ev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="?", default="predictions.json")
    parser.add_argument(
        "--examples", type=int, default=0,
        help="show N before/after examples that fine-tuning fixed",
    )
    args = parser.parse_args()

    with open(args.predictions) as f:
        records = json.load(f)

    base_em, base_ev = score(records, "baseline")
    ft_em, ft_ev = score(records, "finetuned")
    gold_ev = sum(executes_ok(r["context"], r["gold"]) for r in records) / len(records)

    print(f"{'metric':<22}{'base model':>12}{'fine-tuned':>12}")
    print("-" * 46)
    print(f"{'exact match':<22}{base_em:>11.1%}{ft_em:>11.1%}")
    print(f"{'execution validity':<22}{base_ev:>11.1%}{ft_ev:>11.1%}")
    print(f"\ngold queries execute OK on {gold_ev:.1%} of schemas (sanity upper bound)")

    fixed = [
        r for r in records
        if not exact_match(r["baseline"], r["gold"]) and exact_match(r["finetuned"], r["gold"])
    ]
    print(f"{len(fixed)}/{len(records)} examples fixed by fine-tuning")

    for r in fixed[: args.examples]:
        print("=" * 80)
        print("Question:  ", r["question"])
        print("Gold:      ", r["gold"])
        print("Base:      ", r["baseline"])
        print("Fine-tuned:", r["finetuned"])


if __name__ == "__main__":
    main()
