import json

# ── Exact copy from taoyds/spider evaluation.py ───────────────────────────────

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
AGG_OPS   = ('none', 'max', 'min', 'count', 'sum', 'avg')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:
        count += len(sql['from']['table_units']) - 1
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units
                  if cond_unit[1] == WHERE_OPS.index('like')])
    return count


def count_component2(sql):
    return len(get_nestedSQL(sql))


def count_others(sql):
    count = 0
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg(
            [unit[1] for unit in sql['orderBy'][1] if unit[1]] +
            [unit[2] for unit in sql['orderBy'][1] if unit[2]]
        )
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1
    if len(sql['select'][1]) > 1:
        count += 1
    if len(sql['where']) > 1:
        count += 1
    if len(sql['groupBy']) > 1:
        count += 1
    return count


class Evaluator:
    def __init__(self):
        self.partial_scores = None

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)
        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluator = Evaluator()

    with open("C:/Users/Aly/Downloads/spider_data/spider_data/dev.json") as f:
        data = json.load(f)

    counts = {"easy": 0, "medium": 0, "hard": 0, "extra": 0}

    for item in data:
        difficulty = evaluator.eval_hardness(item["sql"])
        item["difficulty"] = difficulty
        counts[difficulty] += 1

    print("Difficulty distribution in Spider dev set:")
    for level, count in counts.items():
        print(f"  {level:<12}: {count}")
    print(f"  Total        : {len(data)}")

    hard_items = [d for d in data if d["difficulty"] in ("hard", "extra")]
    print(f"\nHard + Extra hard: {len(hard_items)} planning questions")

    print("\n3 sample hard questions:")
    for item in [d for d in data if d["difficulty"] == "hard"][:3]:
        print(f"  [{item['db_id']}] {item['question']}")
        print(f"  SQL: {item['query']}")
        print()

    print("3 sample extra hard questions:")
    for item in [d for d in data if d["difficulty"] == "extra"][:3]:
        print(f"  [{item['db_id']}] {item['question']}")
        print(f"  SQL: {item['query']}")
        print()

    with open(
        "C:/Users/Aly/Downloads/spider_data/spider_data/dev_with_difficulty.json", "w"
    ) as f:
        json.dump(data, f, indent=2)

    print("Saved: dev_with_difficulty.json")