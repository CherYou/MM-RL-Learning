"""Model-generated rollouts. Observations are context and have loss_mask=0."""

from .data import ROOT
from .models import Sample
from .environments import LocalSearch, python_tool, extract_tag, make_household, canonical_action
from .rewards import reward

MATH_SYSTEM = "Reason step by step. Put the final answer in <answer>...</answer>."
AGENT_SYSTEMS = {
    "search-r1": "Use <search>query</search> to retrieve evidence. Finish with <answer>answer</answer>.",
    "retool": "Use <python>Python numerical code with print</python> to calculate. Finish with <answer>answer</answer>.",
    "alfworld": "Act in the household environment. Return exactly one <action>admissible action</action> per turn.",
}


def single_rollout(policy, row, config, greedy=False, system=None):
    ids, extras = policy.prompt(
        row["prompt"], system or MATH_SYSTEM, ROOT / row["image"] if row.get("image") else None
    )
    out = policy.generate(ids, config["max_new_tokens"], extras, greedy=greedy)
    text = policy.tokenizer.decode(out, skip_special_tokens=True)
    sample = Sample(
        ids + out,
        [0.0] * len(ids) + [1.0] * len(out),
        len(ids),
        text=text,
        turns=[(len(ids), len(ids) + len(out))],
        extras=extras,
        metadata={"id": row["id"]},
    )
    sample.reward = reward(sample, row, config.get("reward", "math"))
    sample.metadata["truncated"] = not out or out[-1] != policy.tokenizer.eos_token_id
    return policy.snapshot(sample)


def agent_rollout(policy, row, config, greedy=False, initial=None, turn_limit=None):
    algo = config["algorithm"]
    household = algo in {"alfworld", "tempo", "agentopsd"}
    env = make_household(config, row) if household else None
    search = LocalSearch(config.get("corpus", "data/search/corpus.jsonl")) if algo == "search-r1" else None
    actions = []
    observations = []
    try:
        first = env.reset() if env else row["prompt"]
        if initial:
            for action, expected in zip(initial["actions"], initial["observations"], strict=True):
                observation, _, _ = env.step(action)
                if observation != expected:
                    raise ValueError("TEMPO replay observation mismatch")
            ids = list(initial["tokens"])
            actions = list(initial["actions"])
            observations = list(initial["observations"])
        else:
            ids, _ = policy.prompt(first, AGENT_SYSTEMS["alfworld" if household else algo])
        mask = [0.0] * len(ids)
        initial_length = len(ids)
        turns = []
        generated = []
        total_reward = 0.0
        done = False
        stop_reason = "macro_boundary" if turn_limit is not None else "turn_limit"
        invalid_actions = 0
        for turn in range(turn_limit or config["max_turns"]):
            if len(ids) + config["max_new_tokens"] > config.get("max_context_tokens", 4096):
                stop_reason = "context_limit"
                break
            out = policy.generate(ids, config["max_new_tokens"], greedy=greedy)
            text = policy.tokenizer.decode(out, skip_special_tokens=True)
            generated.append(text)
            start = len(ids)
            ids.extend(out)
            mask.extend([1.0] * len(out))
            turns.append((start, len(ids)))
            if env:
                parsed_action = extract_tag(text, "action")
                invalid_actions += parsed_action is None
                action = canonical_action(parsed_action or "invalid action")
                observation, r, done = env.step(action)
                actions.append(action)
                observations.append(observation)
                total_reward = max(total_reward, r)
            elif extract_tag(text, "answer") is not None:
                done = True
                stop_reason = "terminal"
                break
            elif search:
                query = extract_tag(text, "search")
                observation = (
                    search.search(query)
                    if query
                    else "ToolError: expected <search>query</search> or <answer>."
                )
            else:
                code = extract_tag(text, "python")
                observation = (
                    python_tool(code) if code else "ToolError: expected <python>code</python> or <answer>."
                )
            if done:
                stop_reason = "terminal"
                break
            # Append exact IDs. Never re-render and retokenize earlier model output.
            obs_ids = policy.encode(
                "\n<observation>"
                + observation[: config.get("max_observation_chars", 2000)]
                + "</observation>\nAssistant: "
            )
            if len(ids) + len(obs_ids) > config.get("max_context_tokens", 4096):
                stop_reason = "context_limit"
                break
            ids.extend(obs_ids)
            mask.extend([0.0] * len(obs_ids))
        sample = Sample(
            ids,
            mask,
            initial_length,
            text="\n".join(generated),
            turns=turns,
            metadata={
                "id": row["id"],
                "actions": actions,
                "observations": observations,
                "done": done,
                "stop_reason": stop_reason,
                "invalid_actions": invalid_actions,
                "environment": config.get("environment", "tools"),
                "truncated": not done,
            },
        )
        sample.reward = total_reward if env else reward(sample, row, config.get("reward", "math"))
        if config.get("reward") == "debug_token":
            sample.reward = reward(sample, row, "debug_token")
        return policy.snapshot(sample)
    finally:
        if env:
            env.close()
