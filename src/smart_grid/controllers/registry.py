"""Names and styles shared by plots, reports and TensorBoard."""
AGENTS = {
    "delay_aware_agent": {"label": "Aware PPO", "variant": "aware", "color": "#0072B2", "marker": "o", "linestyle": "-"},
    "delay_unaware_agent": {"label": "No-Age PPO", "variant": "no_age", "color": "#D55E00", "marker": "s", "linestyle": "--"},
    "zero_delay_agent": {"label": "Zero-Delay PPO", "variant": "zero_delay", "color": "#8A5BA8", "marker": "^", "linestyle": "-."},
    "rule_based": {"label": "Rule-Based", "variant": "rule_based", "color": "#008566", "marker": "D", "linestyle": (0, (5, 2, 1, 2))},
    "idle_agent": {"label": "Idle", "variant": "idle", "color": "#707070", "marker": "x", "linestyle": ":"},
}
ORDER = tuple(AGENTS)
LEARNED = tuple(key for key in ORDER if AGENTS[key]["variant"] in {"aware", "no_age", "zero_delay"})


def agent_id(name):
    if name in AGENTS:
        return name
    for key, details in AGENTS.items():
        variant = details["variant"]
        if name == variant or name.startswith(variant + "_s"):
            return key
    raise ValueError(f"Unknown controller: {name}")


def require_five(frame):
    present = set(frame["controller"])
    if present != set(ORDER):
        raise ValueError(f"All five controllers are required. Missing: {set(ORDER)-present}; unexpected: {present-set(ORDER)}")

