"""项目统一模型配置，避免各训练脚本重复定义结构参数。"""

PROFILES = {
    # reason_768.pth 对应的项目主线结构。
    "reason_vlm_109m": {
        "hidden_size": 768,
        "num_hidden_layers": 16,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 2048,
        "vocab_size": 6400,
        "qk_norm": False,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1e6,
    },
}


def profile_names():
    return sorted(PROFILES)


def build_vlm_config(profile_name, max_seq_len, use_moe=False):
    """用同一个 profile 构造所有训练阶段的 VLMConfig。"""
    if profile_name not in PROFILES:
        raise ValueError(f"未知 model profile: {profile_name}; 可选: {profile_names()}")
    from model.model_vlm import VLMConfig

    values = dict(PROFILES[profile_name])
    values.update(max_seq_len=max_seq_len, use_moe=bool(use_moe))
    config = VLMConfig(**values)
    config.model_profile = profile_name
    return config


def add_model_profile_argument(parser, default="reason_vlm_109m"):
    parser.add_argument(
        "--model_profile",
        default=default,
        choices=profile_names(),
        help="统一模型结构配置",
    )
