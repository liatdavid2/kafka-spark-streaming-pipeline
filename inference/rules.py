from rules_loader import load_all_rules


def _num(v):
    try:
        return float(v)
    except:
        return 0.0


def check_condition(value, op, target):
    if op == ">":
        return value > target
    if op == "<":
        return value < target
    if op == ">=":
        return value >= target
    if op == "<=":
        return value <= target
    if op == "==":
        return value == target
    return False


def evaluate_rules(flow):
    rules = load_all_rules()

    matched_rules = []
    attack_types = []
    reasons = []
    actions = []
    total_score = 0.0

    for rule in rules:
        results = []

        for cond in rule["conditions"]:
            value = _num(flow.get(cond["field"]))
            results.append(check_condition(value, cond["op"], cond["value"]))

        if rule.get("logic", "AND") == "AND":
            triggered = all(results)
        else:
            triggered = any(results)

        if triggered:
            matched_rules.append(rule["name"])
            attack_types.append(rule["name"])
            reasons.append(rule["description"])
            actions.append(rule.get("action", "ALERT"))
            total_score += rule.get("score", 0.1)

    total_score = min(total_score, 1.0)

    return {
        "rule_score": round(total_score, 4),
        "matched_rules": matched_rules,
        "attack_hypothesis": list(set(attack_types)),
        "reasons": reasons,
        "rule_actions": actions
    }