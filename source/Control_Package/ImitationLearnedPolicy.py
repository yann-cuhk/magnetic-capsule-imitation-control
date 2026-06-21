import os

import numpy as np


ILC_STATE_SIZE = 38
TRACKING_STATE_SIZE = 12
CONTROL_SIZE = 6

DEFAULT_POLICY_PATH = "Control_Package/ilc_policy.npz"


class ImitationLearnedPolicy:
    """Runtime inference for the Imitation-learned control policy."""

    def __init__(self, control_info):
        self.path = control_info.get("policy_path", DEFAULT_POLICY_PATH)
        self.last_mode = "ilc"

        max_force = float(control_info.get("max_force", 0.0015))
        max_moment = float(control_info.get("max_moment", 8e-4))
        self.action_min = np.array(
            [-max_force, -max_force, -max_force, -max_moment, -max_moment, -max_moment],
            dtype=float,
        )
        self.action_max = -self.action_min

        self.seed_position_gain = float(control_info.get("seed_position_gain", 0.0))
        self.seed_attitude_gain = float(control_info.get("seed_attitude_gain", 0.0))
        self.seed_linear_damping = float(control_info.get("seed_linear_damping", 0.0))
        self.seed_angular_damping = float(control_info.get("seed_angular_damping", 0.0))
        self.seed_clip_to_policy_bounds = bool(control_info.get("seed_clip_to_policy_bounds", False))

        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Imitation-learned control policy not found: {self.path}."
            )
        self._load_policy(self.path)

    def _load_policy(self, path):
        data = np.load(path, allow_pickle=True)
        metadata_keys = {"u_min", "u_max", "policy_type"}
        gain_keys = [key for key in data.files if key not in metadata_keys]
        if len(gain_keys) != 1:
            raise KeyError(f"Policy file must contain exactly one gain matrix: {path}")
        K = data[gain_keys[0]]
        self.K_ilc = np.asarray(K, dtype=float).reshape(CONTROL_SIZE, TRACKING_STATE_SIZE)
        self.u_min = np.asarray(data.get("u_min", self.action_min), dtype=float).reshape(CONTROL_SIZE)
        self.u_max = np.asarray(data.get("u_max", self.action_max), dtype=float).reshape(CONTROL_SIZE)
        self.u_min = np.maximum(self.u_min, self.action_min)
        self.u_max = np.minimum(self.u_max, self.action_max)

    def control(self, state):
        state = np.asarray(state, dtype=float).reshape(ILC_STATE_SIZE)
        tracking_state = state[:TRACKING_STATE_SIZE]

        action = self.K_ilc @ tracking_state
        action[:3] += (
            self.seed_position_gain * tracking_state[:3]
            - self.seed_linear_damping * tracking_state[6:9]
        )
        action[3:] += (
            self.seed_attitude_gain * tracking_state[3:6]
            - self.seed_angular_damping * tracking_state[9:12]
        )

        self.last_mode = "ilc"
        if self.seed_clip_to_policy_bounds:
            return np.clip(action, self.u_min, self.u_max)
        return np.clip(action, self.action_min, self.action_max)
