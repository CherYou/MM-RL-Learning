"""Compose the installed verl schema instead of maintaining a stale copied YAML."""

from pathlib import Path
import verl
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


def make_gpu_config(lab):
    config_dir = Path(verl.__file__).parent / "trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=["actor_rollout_ref.rollout.name=vllm"])
    workers = lab.get("verl_workers", 1)
    with open_dict(config):
        config.trainer.n_gpus_per_node = lab.get("gpus_per_node", workers // lab.get("verl_nodes", 1))
        config.trainer.nnodes = lab.get("verl_nodes", 1)
        config.trainer.total_training_steps = lab["steps"]
        config.trainer.logger = ["console", "tensorboard"]
        config.data.train_batch_size = lab["batch_size"]
        config.data.max_prompt_length = lab.get("max_context_tokens", 4096) - lab["max_new_tokens"]
        config.data.max_response_length = lab["max_new_tokens"]
        ar = config.actor_rollout_ref
        ar.model.path = lab["model"]
        ar.model.trust_remote_code = False
        ar.model.use_remove_padding = False
        ar.model.use_fused_kernels = False
        ar.model.override_config = {"attn_implementation": "sdpa"}
        ar.model.enable_gradient_checkpointing = False  # updates use eval mode for exact dropout parity
        ar.model.lora_rank = lab.get("lora_rank", 0)
        ar.actor.strategy = "fsdp"
        ar.actor.use_torch_compile = False
        ar.actor.use_rollout_log_probs = True
        ar.actor.fsdp_config.use_orig_params = False
        ar.actor.fsdp_config.param_offload = True
        ar.actor.fsdp_config.optimizer_offload = True
        ar.actor.fsdp_config.dtype = "bfloat16"
        ar.actor.ppo_mini_batch_size = lab["batch_size"] * lab["group_size"]
        ar.actor.ppo_micro_batch_size_per_gpu = lab.get("micro_batch_size", 1)
        ar.actor.ppo_epochs = lab.get("ppo_epochs", 1)
        ar.actor.grad_clip = lab.get("max_grad_norm", 1.0)
        ar.actor.optim.lr = lab["learning_rate"]
        ar.actor.optim.weight_decay = 0.0
        ar.actor.optim.total_training_steps = lab["steps"]
        ar.actor.checkpoint.save_contents = ["model", "optimizer", "extra", "hf_model"]
        ar.actor.checkpoint.load_contents = ["model", "optimizer", "extra"]
        ar.ref.fsdp_config.param_offload = True
        ar.ref.log_prob_micro_batch_size_per_gpu = lab.get("micro_batch_size", 1)
        ar.rollout.name = lab.get("rollout_engine", "vllm")
        ar.rollout.mode = "async"
        ar.rollout.n = 1
        ar.rollout.engine_kwargs.vllm.seed = lab.get("seed", 42)
        ar.rollout.temperature = 1.0
        ar.rollout.top_p = 1.0
        ar.rollout.top_k = -1
        ar.rollout.prompt_length = config.data.max_prompt_length
        ar.rollout.response_length = lab["max_new_tokens"]
        ar.rollout.max_model_len = lab.get("max_context_tokens", 4096)
        ar.rollout.tensor_model_parallel_size = lab.get("rollout_tensor_parallel_size", 1)
        ar.rollout.gpu_memory_utilization = lab.get("rollout_gpu_memory_utilization", 0.5)
        ar.rollout.calculate_log_probs = True
        ar.rollout.log_prob_micro_batch_size_per_gpu = lab.get("micro_batch_size", 1)
        ar.rollout.enforce_eager = True
        ar.rollout.free_cache_engine = True
        ar.rollout.agent.num_workers = 0  # LabAgentLoop supplies environment workers.
        config.critic.model.path = lab.get("critic_model", lab["model"])
        config.critic.model.override_config = {"attn_implementation": "sdpa"}
        config.critic.ppo_mini_batch_size = lab["batch_size"] * lab["group_size"]
        config.critic.ppo_micro_batch_size_per_gpu = lab.get("micro_batch_size", 1)
        config.critic.optim.lr = lab["learning_rate"]
        config.critic.optim.total_training_steps = lab["steps"]
        config.critic.model.fsdp_config.param_offload = True
        config.critic.model.fsdp_config.optimizer_offload = True
        config.critic.checkpoint.save_contents = ["model", "optimizer", "extra"]
        config.lab = lab
    OmegaConf.resolve(config)
    return config
