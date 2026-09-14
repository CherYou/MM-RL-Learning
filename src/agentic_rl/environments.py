"""Local retrieval, restricted Python tools and a real ALFWorld adapter."""

from collections import Counter
import ast
import json
import math
import re
import subprocess
import sys
import tempfile

from .data import ROOT, read_jsonl


def canonical_action(text):
    return " ".join(text.strip().lower().split())


class LocalSearch:
    def __init__(self, corpus="data/search/corpus.jsonl"):
        self.docs = read_jsonl(corpus)
        self.counts = [Counter(self.tokens(d["title"] + " " + d["text"])) for d in self.docs]
        self.df = Counter(t for counts in self.counts for t in counts)
        self.avg_len = sum(map(lambda x: sum(x.values()), self.counts)) / len(self.counts)

    @staticmethod
    def tokens(text):
        return re.findall(r"\w+", text.lower())

    def search(self, query, k=3):
        terms = self.tokens(query)

        def score(i):
            tf = self.counts[i]
            length = sum(tf.values())
            return sum(
                math.log(1 + (len(self.docs) - self.df[t] + 0.5) / (self.df[t] + 0.5))
                * tf[t]
                * 2.2
                / (tf[t] + 1.2 * (0.25 + 0.75 * length / max(1, self.avg_len)))
                for t in terms
            )

        ranked = sorted(range(len(self.docs)), key=score, reverse=True)[:k]
        return (
            "\n".join(f"[{self.docs[i]['title']}] {self.docs[i]['text']}" for i in ranked if score(i) > 0)
            or "No documents found."
        )


def _limits():
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def python_tool(code, timeout=3):
    """Execute a bounded numerical Python subset in a separate process.

    No file/network/import operations or arbitrary attributes; see docs/TOOLS.md.
    """
    if len(code) > 16000:
        return "ToolError: code exceeds 16K characters"
    builtins = {
        "print",
        "sum",
        "range",
        "len",
        "min",
        "max",
        "abs",
        "round",
        "int",
        "float",
        "list",
        "sorted",
        "enumerate",
        "zip",
    }
    math_functions = {
        "sqrt",
        "sin",
        "cos",
        "tan",
        "log",
        "log10",
        "exp",
        "floor",
        "ceil",
        "factorial",
        "gcd",
        "comb",
        "pi",
        "e",
    }
    allowed = (
        ast.Module,
        ast.Expr,
        ast.Assign,
        ast.AugAssign,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.keyword,
        ast.List,
        ast.Tuple,
        ast.Subscript,
        ast.Slice,
        ast.For,
        ast.If,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Not,
        ast.IfExp,
        ast.Attribute,
        ast.Import,
        ast.alias,
        ast.ListComp,
        ast.comprehension,
    )
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ValueError(f"Unsupported syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise ValueError("Private names are unavailable")
            if isinstance(node, ast.Import) and any(a.name != "math" or a.asname for a in node.names):
                raise ValueError("Only 'import math' is available")
            if isinstance(node, ast.Attribute) and not (
                isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr in math_functions
            ):
                raise ValueError("Only selected math attributes are available")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id not in builtins
            ):
                raise ValueError(f"Unavailable function: {node.func.id}")
        runner = (
            "import math,json,sys\nc=json.loads(sys.stdin.read())\nb={k:__builtins__.__dict__[k] for k in "
            + repr(sorted(builtins))
            + "}\nexec(compile(c.replace('import math',''),'tool','exec'),{'__builtins__':b,'math':math})"
        )
        with tempfile.TemporaryDirectory(prefix="arl-python-") as cwd:
            with tempfile.TemporaryFile(mode="w+") as output:
                result = subprocess.run(
                    [sys.executable, "-I", "-S", "-c", runner],
                    input=json.dumps(code),
                    text=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    env={"PATH": "/usr/bin:/bin"},
                    timeout=timeout,
                    preexec_fn=_limits,
                )
                output.seek(0)
                text = output.read(8192)
        return text.strip() if result.returncode == 0 else "ToolError: " + text.strip()
    except (SyntaxError, ValueError, subprocess.TimeoutExpired) as e:
        return f"ToolError: {e}"


class ToyHousehold:
    """Named fixture for CPU tests, never reported as ALFWorld."""

    def __init__(self):
        self.history = []
        self.held = False
        self.won = False

    def reset(self):
        self.history = []
        self.held = False
        self.won = False
        return "Your task is to: put the apple in the fridge. Apple on table. Fridge open."

    def step(self, action):
        self.history.append(action)
        if action == "take apple from table":
            self.held = True
            return "You pick up the apple.", 0.0, False
        if action == "put apple in fridge" and self.held:
            self.won = True
            return "Task completed.", 1.0, True
        return "Nothing happens.", 0.0, False

    def close(self):
        pass


class ALFWorld:
    def __init__(self, game, max_turns=50, seed=42):
        from .alfworld_env import ALFWorldGroup

        self.env = ALFWorldGroup(game, 1, max_turns, seed=seed, asynchronous=False)
        self.history = []

    def reset(self):
        state = self.env.reset()[0]
        self.history = []
        return state.observation + "\nAdmissible actions: " + "; ".join(state.admissible_actions)

    def step(self, action):
        self.history.append(action)
        s = self.env.step([action])[0]
        return (
            s.observation + "\nAdmissible actions: " + "; ".join(s.admissible_actions),
            float(s.won),
            s.done,
        )

    def close(self):
        self.env.close()


def extract_tag(text, tag):
    match = re.search(r"<" + tag + r">(.*?)</" + tag + r">", text, re.S)
    return match.group(1).strip() if match else None


def make_household(config, row):
    if config.get("environment") == "toy":
        return ToyHousehold()
    from .alfworld_data import GameExample

    return ALFWorld(
        GameExample(row["id"], row["split"], ROOT / row["game_file"], row["task_type"]),
        config.get("max_turns", 50),
        config.get("seed", 42),
    )
