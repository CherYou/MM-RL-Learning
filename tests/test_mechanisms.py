import copy
import json
import math
import pytest
import torch

from agentic_rl.losses import policy_loss, group_advantages, opd_loss, dpo_loss, gae, overlong_penalty
from agentic_rl.models import Policy, VisionPolicy, Sample
from agentic_rl.rollout import single_rollout, agent_rollout
from agentic_rl.rewards import exact_match, extract_answer, token_f1
from agentic_rl.harness import capture, prefix_trees, CAPORouter
from agentic_rl.agentopsd import reshape_turn_advantages
from agentic_rl.tempo import signals, parse_value
from agentic_rl.environments import python_tool, LocalSearch
from agentic_rl.data import ROOT, read_jsonl, report_path, write_jsonl

torch.set_num_threads(2)


@pytest.fixture
def policy():
    return Policy(seed=12)


def test_constant_groups_stay_zero():
    adv = group_advantages(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]), 4)
    assert torch.equal(adv[:4], torch.zeros(4))
    assert adv[4:].mean() == 0
    assert torch.isfinite(adv).all()


def test_prompt_and_observation_gradients_are_zero():
    lp = torch.zeros((2, 4), requires_grad=True)
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    loss, _ = policy_loss(lp, lp.detach(), torch.tensor([1.0, -1.0]), mask)
    loss.backward()
    assert torch.equal(lp.grad[mask == 0], torch.zeros_like(lp.grad[mask == 0]))
    assert lp.grad[mask != 0].abs().sum() > 0


@pytest.mark.parametrize(
    "adv,ratio,clipped", [(1.0, 1.5, True), (-1.0, 0.5, True), (1.0, 0.5, False), (-1.0, 1.5, False)]
)
def test_ppo_clipping_direction(adv, ratio, clipped):
    lp = torch.tensor([[math.log(ratio)]], requires_grad=True)
    loss, _ = policy_loss(lp, torch.zeros_like(lp), torch.tensor([adv]), torch.ones_like(lp))
    loss.backward()
    assert (float(lp.grad) == 0) == clipped


def test_cispo_keeps_gradient_beyond_clip():
    lp = torch.tensor([[math.log(2.0)]], requires_grad=True)
    loss, _ = policy_loss(lp, torch.zeros_like(lp), torch.tensor([1.0]), torch.ones_like(lp), kind="cispo")
    loss.backward()
    assert float(lp.grad) == pytest.approx(-1.2)


def test_gspo_geometric_mean_not_product():
    lp = torch.tensor([[math.log(2.0), math.log(0.5), 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    _, stats = policy_loss(lp, torch.zeros_like(lp), torch.tensor([1.0]), mask, kind="gspo")
    assert float(stats["policy/ratio"]) == pytest.approx(1.0)


def test_dapo_token_normalization():
    lp = torch.zeros((2, 3), requires_grad=True)
    mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    grpo, _ = policy_loss(lp, lp.detach(), torch.tensor([1.0, -1.0]), mask, kind="grpo")
    dapo, _ = policy_loss(lp, lp.detach(), torch.tensor([1.0, -1.0]), mask, kind="dapo")
    assert float(grpo.detach()) == 0
    assert float(dapo.detach()) == pytest.approx(0.5)
    assert overlong_penalty(torch.tensor([5, 7, 10]), 5, 10).tolist() == pytest.approx([0.0, -0.4, -1.0])


def test_sampled_reverse_kl_gradient_and_frozen_teacher():
    p = torch.tensor([0.3, 0.7], requires_grad=True)
    logp = p.log()[None]
    teacher = torch.tensor([[math.log(0.6), math.log(0.4)]], requires_grad=True)
    # Enumerate both categorical outcomes, weight by on-policy sampling probability.
    gap = logp.detach() - teacher.detach()
    loss = ((logp - logp.detach()).exp() * gap * p.detach()).sum()
    loss.backward()
    assert torch.allclose(p.grad, gap[0])
    assert teacher.grad is None
    lp = torch.zeros((1, 2), requires_grad=True)
    actual, _ = opd_loss(lp, lp.detach(), torch.tensor([[1.0, -1.0]]), torch.ones_like(lp))
    actual.backward()
    assert lp.grad.tolist()[0] == pytest.approx([-0.5, 0.5])


def test_dpo_moves_chosen_up_rejected_down():
    chosen = torch.tensor([0.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    loss, _ = dpo_loss(chosen, rejected, torch.tensor([0.0]), torch.tensor([0.0]))
    loss.backward()
    assert chosen.grad < 0 and rejected.grad > 0
    assert float(loss) == pytest.approx(math.log(2))


def test_gae_terminal_and_truncation():
    rewards = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.tensor([[0.2, 0.4, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv, ret = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert ret.tolist()[0] == pytest.approx([1.0, 1.0, 0.0])
    _, ret = gae(
        torch.zeros((1, 2)), torch.zeros((1, 2)), torch.ones((1, 2)), lam=1.0, bootstrap=torch.tensor([0.7])
    )
    assert ret.tolist()[0] == pytest.approx([0.7, 0.7])


def test_tempo_bootstrap_and_strict_value():
    ret, adv, target = signals([1.0, 0.0], [0.9, 0.4], [True, False])
    assert ret.tolist() == pytest.approx([1.0, 0.4])
    assert float(target) == pytest.approx(0.7)
    assert adv.sum().abs() < 1e-6
    assert parse_value("<value>0.7</value>") == 0.7
    assert parse_value("<value>7</value>") is None
    assert parse_value("<value>0.1</value><value>0.2</value>") is None


def test_agentopsd_bounded_credit_preserves_sign():
    for advantage in [-1.0, 0.0, 1.0]:
        credit = reshape_turn_advantages(advantage, 0.5, [10.0, -10.0, 2.0])
        for turn in credit.turns:
            assert 0.9 * abs(advantage) - 1e-7 <= abs(turn.advantage) <= 1.1 * abs(advantage) + 1e-7
            assert turn.advantage * advantage >= 0


def test_numeric_and_boxed_reward():
    assert exact_match("<answer>1,234</answer>", "1234") == 1
    assert extract_answer(r"work \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert token_f1("the red apple", "red apple") == 1
    assert exact_match("", "") == 1
    assert exact_match(r"\boxed{\frac{1}{2}}", "0.5") == 1
    assert exact_match("I think 4 and maybe 3", "3") == 0


def test_tool_exec_and_boundaries():
    assert python_tool("import math\nprint(math.sqrt(81) + 1)") == "10.0"
    assert python_tool("print(sum(range(10)))") == "45"
    assert python_tool("open('forbidden.txt').read()").startswith("ToolError")
    assert python_tool("import os").startswith("ToolError")


def test_jsonl_embedded_unicode_line_separator(tmp_path):
    p = tmp_path / "data.jsonl"
    rows = [{"prompt": "x\u2028y\u0085z", "answer": "1"}]
    write_jsonl(p, rows)
    assert read_jsonl(p) == rows


def test_report_path_sanitizes_native_and_foreign_absolute_paths():
    assert report_path(ROOT / "models/example") == "models/example"
    assert report_path("models/example") == "models/example"
    assert report_path("/private/mount/models/example") == "<external>/example"
    windows_path = "C:" + chr(92) + chr(92).join(["Users", "alice", "models", "example"])
    assert report_path(windows_path) == "<external>/example"


def test_token_snapshot_and_reload(policy, tmp_path):
    row = {"id": "x", "prompt": "What is 1+2?", "answer": "3"}
    s = single_rollout(policy, row, {"max_new_tokens": 8})
    lp, m, _ = policy.score([s])
    assert torch.allclose(lp[0], torch.tensor(s.old_logp), atol=1e-6)
    assert m.sum() == sum(s.mask)
    policy.save(tmp_path / "checkpoint")
    restored = Policy(str(tmp_path / "checkpoint/model"))
    assert torch.allclose(lp, restored.score([s])[0], atol=1e-6)


def test_scripted_tool_rollout_excludes_observation(policy, monkeypatch):
    outputs = iter([policy.encode("<python>print(2+3)</python>"), policy.encode("<answer>5</answer>")])
    monkeypatch.setattr(policy, "generate", lambda *a, **k: next(outputs))
    s = agent_rollout(
        policy,
        {"id": "x", "prompt": "What is 2+3?", "answer": "5"},
        {"algorithm": "retool", "max_new_tokens": 50, "max_turns": 2, "max_context_tokens": 4096},
    )
    assert s.reward == 1
    assert len(s.turns) == 2
    assert not any(s.mask[s.turns[0][1] : s.turns[1][0]])
    assert sum(s.mask) == sum(end - start for start, end in s.turns)


def test_macro_state_replay_restores_environment_and_masks_old_prefix(policy, monkeypatch):
    from agentic_rl.tempo import StateStore

    outputs = iter(
        [
            policy.encode("<action>take apple from table</action>"),
            policy.encode("<action>put apple in fridge</action>"),
        ]
    )
    monkeypatch.setattr(policy, "generate", lambda *a, **k: next(outputs))
    row = {"id": "toy", "prompt": "household", "answer": "success"}
    config = {
        "algorithm": "tempo",
        "environment": "toy",
        "max_new_tokens": 64,
        "max_turns": 4,
        "max_context_tokens": 4096,
    }
    first = agent_rollout(policy, row, config, turn_limit=1)
    assert first.metadata["stop_reason"] == "macro_boundary"
    store = StateStore()
    store.add(row, first)
    state = store.pop()
    second = agent_rollout(policy, row, config, initial=state, turn_limit=1)
    assert second.reward == 1 and second.metadata["done"]
    assert not any(second.mask[: len(first.tokens)])
    store.add(row, second)
    assert not store.states


def test_session_tree_keeps_branches_and_calls(policy):
    a = capture(
        policy,
        [1, 5],
        policy.encode('{"action":"Reason","args":"x"}'),
        rollout_id="r",
        session_id="a",
        call_id="1",
    )
    b = capture(
        policy,
        [1, 6],
        policy.encode('{"action":"Summary","args":"y"}'),
        rollout_id="r",
        session_id="a",
        call_id="2",
    )
    c = copy.deepcopy(a)
    c.session_id = "b"
    c.call_id = "3"
    trees = prefix_trees([a, b, c])
    assert len(trees) == 2
    tree = trees[("r", "a")]
    node = tree["nodes"][0]["children"][1]
    assert len(tree["nodes"][node]["children"]) == 2
    assert sum(len(n["outputs"]) for n in tree["nodes"]) == 2


def test_capo_routes_actual_gradients(policy):
    records = []
    for i, (action, arg) in enumerate([("Calculate", "print(2+3)"), ("Summary", "5")]):
        ids = policy.encode("Question: 2+3? Assistant: ")
        r = capture(
            policy,
            ids,
            policy.encode(json.dumps({"action": action, "args": arg})),
            rollout_id=str(i),
            session_id="central",
            call_id="0",
        )
        r.sample.reward = 1.0
        records.append(r)
        assert not any(a * g for a, g in zip(r.action_mask, r.args_mask))
    router = CAPORouter(policy)
    router.probe(records)
    lp, m, _ = policy.score([r.sample for r in records])

    def mask(field):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(getattr(r, field)[1:]) for r in records], batch_first=True
        )

    am = mask("action_mask")
    gm = mask("args_mask")
    stats = router.backward(-(lp * am).sum(), (lp * gm).sum())
    assert stats["capo/routed_parameters"] > 0
    gradient = 0.0
    for name, p in policy.lm.named_parameters():
        if name not in router.masks:
            assert p.grad is None
        else:
            union = sum(router.masks[name]) > 0
            assert p.grad[~union].abs().sum() == 0
            gradient += float(p.grad.abs().sum())
    assert gradient > 0


def test_images_change_logprobs_and_receive_gradient(tmp_path):
    from PIL import Image

    images = []
    for i, color in enumerate(["red", "blue"]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (16, 16), color).save(path)
        images.append(path)
    p = VisionPolicy(seed=3)
    samples = []
    for image in images:
        ids, extras = p.prompt("Choose A or B", image_path=image)
        out = p.encode("A")
        samples.append(Sample(ids + out, [0.0] * len(ids) + [1.0] * len(out), len(ids), extras=extras))
    lp, m, _ = p.score(samples)
    assert not torch.allclose(lp[0, -1], lp[1, -1], atol=1e-7)
    (-lp * m).sum().backward()
    assert any(
        t.grad is not None and t.grad.abs().sum() > 0
        for n, t in p.named_parameters()
        if "multi_modal_projector" in n
    )
    p.save(tmp_path / "vision")
    loaded = VisionPolicy(str(tmp_path / "vision/model"))
    assert torch.allclose(lp, loaded.score(samples)[0], atol=1e-6)


def test_vision_sampling_and_scoring_use_same_support(tmp_path):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "green").save(path)
    policy = VisionPolicy(seed=10)
    ids, extras = policy.prompt("Choose A or B.", image_path=path)
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        generated = policy.lm.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=6,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            suppress_tokens=policy.blocked_output_ids,
            return_dict_in_generate=True,
            output_scores=True,
            **extras,
        )
    output = generated.sequences[0, len(ids) :].tolist()
    assert not set(output) & set(policy.blocked_output_ids)
    sample = Sample(ids + output, [0.0] * len(ids) + [1.0] * len(output), len(ids), extras=extras)
    logp, mask, _ = policy.score([sample])
    sampling = torch.stack(
        [scores[0].log_softmax(-1)[token] for scores, token in zip(generated.scores, output)]
    )
    assert torch.isfinite(logp).all()
    assert torch.allclose(logp[0, len(ids) - 1 :], sampling, atol=1e-5)


def test_real_alfworld_and_retrieval():
    from agentic_rl.alfworld_data import discover_games
    from agentic_rl.environments import ALFWorld

    game = discover_games(ROOT / "data/alfworld", "train")[0]
    env = ALFWorld(game)
    try:
        start = env.reset()
        observed, _, _ = env.step("look")
        assert "Your task is to:" in start
        assert isinstance(observed, str) and observed
    finally:
        env.close()
    search = LocalSearch()
    doc = search.docs[0]
    assert search.search(doc["title"]) != "No documents found."
