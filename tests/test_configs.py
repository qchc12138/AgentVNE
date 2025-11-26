from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from tests.test_strategy import TestConfig

DEFAULT_SN_TOPOLOGY = "/home/yc2/mrt/a/topo/SN_topology.json"
DEFAULT_WORKFLOW_TYPES = {
    "workflow1": "/home/yc2/mrt/a/workflow_topo/workflow1_topo.json",
}
DEFAULT_PARAMETER_SETS = [
    {"arrival_rate": 0.2, "mean_lifetime": 20.0, "max_time_steps": 100, "seed": 2025},
]

# 常用策略的默认参数（可在 smoke test 中复用）
SMOKE_PARAMETER_SETS = {
    "small_basic": {"arrival_rate": 0.2, "mean_lifetime": 20.0, "max_time_steps": 100, "seed": 2025},
}


@dataclass
class ParameterSpec:
    arrival_rate: float
    mean_lifetime: float
    max_time_steps: int
    seed: int
    device: Optional[str] = None
    penalty: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ParameterSpec":
        return cls(
            arrival_rate=float(data["arrival_rate"]),
            mean_lifetime=float(data["mean_lifetime"]),
            max_time_steps=int(data.get("max_time_steps", 100)),
            seed=int(data.get("seed", 2025)),
            device=data.get("device"),
            penalty=float(data["penalty"]) if "penalty" in data else None,
        )

    @classmethod
    def from_string(cls, text: str) -> "ParameterSpec":
        parts: Dict[str, str] = {}
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"参数 '{item}' 缺少 '=' 分隔符")
            key, value = item.split("=", 1)
            parts[key.strip()] = value.strip()
        try:
            parts["arrival_rate"] = float(parts.get("arrival_rate", parts.get("arrival")))
            parts["mean_lifetime"] = float(parts.get("mean_lifetime", parts.get("mean")))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "必须指定 arrival_rate 与 mean_lifetime，格式示例：arrival_rate=0.5,mean_lifetime=50"
            ) from exc
        if "max_time_steps" in parts:
            parts["max_time_steps"] = int(parts["max_time_steps"])
        if "seed" in parts:
            parts["seed"] = int(parts["seed"])
        if "penalty" in parts:
            parts["penalty"] = float(parts["penalty"])
        return cls.from_dict(parts)  # type: ignore[arg-type]


def parse_workflows(values: Optional[Iterable[str]]) -> Dict[str, str]:
    if not values:
        return dict(DEFAULT_WORKFLOW_TYPES)
    result: Dict[str, str] = {}
    for entry in values:
        if "=" not in entry:
            raise ValueError(f"workflow 参数 '{entry}' 缺少 '='")
        name, path = entry.split("=", 1)
        result[name.strip()] = path.strip()
    if not result:
        raise ValueError("至少需要一个 workflow 配置")
    return result


def parse_parameters(values: Optional[Iterable[str]]) -> List[ParameterSpec]:
    if not values:
        return [ParameterSpec.from_dict(item) for item in DEFAULT_PARAMETER_SETS]
    return [ParameterSpec.from_string(item) for item in values]


def get_smoke_config(
    *,
    preset: str = "small_basic",
    overrides: Optional[Dict[str, float]] = None,
    sn_topology: Optional[str] = None,
    workflows: Optional[Dict[str, str]] = None,
    device: str = "cpu",
    penalty: float = -150.0,
) -> TestConfig:
    """
    构造常用的 smoke 测试配置。
    """

    base = dict(SMOKE_PARAMETER_SETS.get(preset, SMOKE_PARAMETER_SETS["small_basic"]))
    if overrides:
        base.update(overrides)
    params = ParameterSpec.from_dict(base)
    return build_config(
        sn_topology=sn_topology or DEFAULT_SN_TOPOLOGY,
        workflows=workflows or dict(DEFAULT_WORKFLOW_TYPES),
        base_device=device,
        base_penalty=penalty,
        params=params,
    )

def build_config(
    *,
    sn_topology: str,
    workflows: Dict[str, str],
    base_device: str,
    base_penalty: float,
    params: ParameterSpec,
) -> TestConfig:
    return TestConfig(
        sn_topology_path=sn_topology,
        workflow_types=workflows,
        arrival_rate=params.arrival_rate,
        mean_lifetime=params.mean_lifetime,
        max_time_steps=params.max_time_steps,
        device=params.device or base_device,
        seed=params.seed,
        penalty=params.penalty if params.penalty is not None else base_penalty,
    )


def build_round_configs(
    *,
    parameter_specs: List[ParameterSpec],
    sn_topology: str,
    workflows: Dict[str, str],
    base_device: str,
    base_penalty: float,
) -> List[TestConfig]:
    if not parameter_specs:
        raise ValueError("至少需要一组测试参数。")

    return [
        build_config(
            sn_topology=sn_topology,
            workflows=workflows,
            base_device=base_device,
            base_penalty=base_penalty,
            params=spec,
        )
        for spec in parameter_specs
    ]

