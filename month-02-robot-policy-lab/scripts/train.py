"""
W2D2 smoke test - confirms draccus resolves conf/pusht_act.json + CLI
overrides correctly. This does not replace lerobot-train - it's a debug
harness so we can inspect the resolved config before trusting a real run.
"""
from robot_policy_lab.utils.reproducibility import seed_everything
import dataclasses

# Importing this triggers ACTConfig's @register_subclass("act") decorator,
# which is what makes --policy.type=act / "type": "act" resolvable at all.
# lerobot_train.py's real entry point calls register_third_party_plugins()
# for the same reason - we're doing the minimal version of that here.
import lerobot.policies.act.configuration_act  # noqa: F401
from lerobot.configs.train import TrainPipelineConfig

from robot_policy_lab.utils.device import configure_mps_env
configure_mps_env()     # must run before draccus parses config and before any tensor creation
seed_everything(cfg.seed)

import draccus

@draccus.wrap()
def main(cfg: TrainPipelineConfig):
    print(dataclasses.asdict(cfg))


if __name__ == "__main__":
    main()