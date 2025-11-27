"""Dueling Double DQN agent for ConnectX.

This module contains the Dueling CNN architecture plus the Double DQN
training loop exactly as defined in the `drl.ipynb` notebook.  It also ships a
Kaggle-ready `my_agent` and helper utilities for exporting trained weights.
"""

from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


DEVICE = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu") if HAS_TORCH else None
EMBEDDED_STATE_DICT: Optional[Dict[str, List]] = None
EMBEDDED_STATE_OUTPUT: Optional[Path] = Path(__file__).resolve().with_name("drl_dueling_state_dict.txt") if "__file__" in globals() else None

GAMMA = 0.98
BUFFER_LIMIT = 50_000
BATCH_SIZE = 32
EPSILON_START = 0.10
EPSILON_END = 0.01
EPSILON_DECAY = 2000
TARGET_SYNC = 200
MIN_REPLAY = 5_000

Transition = Tuple[np.ndarray, int, float, np.ndarray, float]


def _get_cfg_value(cfg, attr: str, default: int) -> int:
    if hasattr(cfg, attr):
        value = getattr(cfg, attr)
    elif isinstance(cfg, dict):
        value = cfg.get(attr, default)
    else:
        value = default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_board(observation) -> List[int]:
    if isinstance(observation, dict):
        board = observation.get("board")
        if board is None:
            nested = observation.get("observation")
            if isinstance(nested, dict):
                board = nested.get("board")
        return board if board is not None else []
    return getattr(observation, "board", [])


def state_from_board(board_flat: List[int], cfg) -> np.ndarray:
    rows = _get_cfg_value(cfg, "rows", 6)
    cols = _get_cfg_value(cfg, "columns", 7)
    if not board_flat:
        board_flat = [0] * (rows * cols)
    grid = np.asarray(board_flat, dtype=np.float32).reshape(rows, cols)
    return grid[np.newaxis, ...]


def find_playable_columns(board_flat: List[int], cfg) -> List[int]:
    cols = _get_cfg_value(cfg, "columns", 7)
    playable = []
    for c in range(cols):
        if c < len(board_flat) and board_flat[c] == 0:
            playable.append(c)
    return playable


def fallback_move(playable: List[int], cfg) -> int:
    if not playable:
        return 0
    center = _get_cfg_value(cfg, "columns", 7) // 2
    return center if center in playable else random.choice(playable)


class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_LIMIT):
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def put(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        s, a, r, sp, dm = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.stack(sp).astype(np.float32),
            np.asarray(dm, dtype=np.float32),
        )


class DuelingQNetwork(nn.Module if HAS_TORCH else object):
    def __init__(self, action_dim: int = 7):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to instantiate DuelingQNetwork.")
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_value = nn.Linear(64 * 2 * 3, 128)
        self.value = nn.Linear(128, 1)
        self.fc_adv = nn.Linear(64 * 2 * 3, 128)
        self.adv = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feat = self.cnn(x)
        v = F.relu(self.fc_value(feat))
        a = F.relu(self.fc_adv(feat))
        v = self.value(v)
        a = self.adv(a)
        return v + a - a.mean(dim=1, keepdim=True)


def epsilon_by_step(step: int) -> float:
    progress = min(1.0, step / EPSILON_DECAY)
    return EPSILON_START - (EPSILON_START - EPSILON_END) * progress


def optimize_double_dueling(
    q_net: DuelingQNetwork,
    target_net: DuelingQNetwork,
    batch: Transition,
    optimizer: torch.optim.Optimizer,
) -> float:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for training.")

    states, actions, rewards, next_states, masks = batch
    states_t = torch.from_numpy(states).to(DEVICE)
    actions_t = torch.from_numpy(actions).long().unsqueeze(-1).to(DEVICE)
    rewards_t = torch.from_numpy(rewards).unsqueeze(-1).to(DEVICE)
    next_states_t = torch.from_numpy(next_states).to(DEVICE)
    masks_t = torch.from_numpy(masks).unsqueeze(-1).to(DEVICE)

    q_values = q_net(states_t).gather(1, actions_t)

    with torch.no_grad():
        next_q_online = q_net(next_states_t)
        next_actions = next_q_online.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states_t)
        next_values = next_q_target.gather(1, next_actions)
        targets = rewards_t + GAMMA * next_values * masks_t

    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_dueling_dqn_agent(episodes: int = 3000) -> DuelingQNetwork:
    if not HAS_TORCH:
        raise ImportError("PyTorch가 설치되어야 학습을 진행할 수 있습니다.")

    from kaggle_environments import make

    base_env = make("connectx", debug=True)
    trainer = base_env.train([None, "random"])
    rows, cols = base_env.configuration.rows, base_env.configuration.columns

    q_net = DuelingQNetwork(cols).to(DEVICE)
    target_net = DuelingQNetwork(cols).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=5e-4)
    buffer = ReplayBuffer()

    global_step = 0
    for episode in range(episodes):
        obs = trainer.reset()
        state = np.array(obs["board"]).reshape(1, rows, cols).astype(np.float32)
        done = False
        while not done:
            epsilon = epsilon_by_step(global_step)
            playable = find_playable_columns(obs["board"], base_env.configuration)
            if random.random() < epsilon:
                action = random.choice(playable)
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                    q_values = q_net(state_t)[0]
                    mask = torch.full_like(q_values, -float("inf"))
                    for c in playable:
                        mask[c] = q_values[c]
                    action = int(mask.argmax().item())

            next_obs, reward, done, _ = trainer.step(int(action))
            next_state = np.array(next_obs["board"]).reshape(1, rows, cols).astype(np.float32)
            buffer.put((state, action, reward, next_state, 0.0 if done else 1.0))

            state = next_state
            obs = next_obs
            global_step += 1

            if len(buffer) >= max(BATCH_SIZE, MIN_REPLAY):
                batch = buffer.sample(BATCH_SIZE)
                optimize_double_dueling(q_net, target_net, batch, optimizer)

        if (episode + 1) % TARGET_SYNC == 0:
            target_net.load_state_dict(q_net.state_dict())

    q_net.eval()
    return q_net.to("cpu")


class EmbeddedPolicy:
    def __init__(self, literal: Dict[str, List]):
        self.w1 = np.asarray(literal["cnn.0.weight"], dtype=np.float32)
        self.b1 = np.asarray(literal["cnn.0.bias"], dtype=np.float32)
        self.w2 = np.asarray(literal["cnn.2.weight"], dtype=np.float32)
        self.b2 = np.asarray(literal["cnn.2.bias"], dtype=np.float32)
        self.fc_value_w = np.asarray(literal["fc_value.weight"], dtype=np.float32)
        self.fc_value_b = np.asarray(literal["fc_value.bias"], dtype=np.float32)
        self.value_w = np.asarray(literal["value.weight"], dtype=np.float32)
        self.value_b = np.asarray(literal["value.bias"], dtype=np.float32)
        self.fc_adv_w = np.asarray(literal["fc_adv.weight"], dtype=np.float32)
        self.fc_adv_b = np.asarray(literal["fc_adv.bias"], dtype=np.float32)
        self.adv_w = np.asarray(literal["adv.weight"], dtype=np.float32)
        self.adv_b = np.asarray(literal["adv.bias"], dtype=np.float32)

    def _conv(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        out_channels, _, kh, kw = weight.shape
        _, h, w = x.shape
        out_h = h - kh + 1
        out_w = w - kw + 1
        out = np.zeros((out_channels, out_h, out_w), dtype=np.float32)
        for oc in range(out_channels):
            for ic in range(weight.shape[1]):
                kernel = weight[oc, ic]
                for i in range(out_h):
                    for j in range(out_w):
                        region = x[ic, i : i + kh, j : j + kw]
                        out[oc, i, j] += np.sum(region * kernel)
            out[oc] += bias[oc]
        return out

    def act(self, state: np.ndarray, playable: List[int]) -> int:
        x = state.astype(np.float32)
        x = self._conv(x, self.w1, self.b1)
        x = np.maximum(x, 0, out=x)
        x = self._conv(x, self.w2, self.b2)
        x = np.maximum(x, 0, out=x)
        flat = x.reshape(-1)

        v_hidden = np.maximum(self.fc_value_w @ flat + self.fc_value_b, 0)
        a_hidden = np.maximum(self.fc_adv_w @ flat + self.fc_adv_b, 0)
        v = (self.value_w @ v_hidden + self.value_b)[0]
        a = self.adv_w @ a_hidden + self.adv_b
        logits = v + a - np.mean(a)

        masked = np.full_like(logits, -np.inf)
        masked[playable] = logits[playable]
        return int(np.argmax(masked))


_INFERENCE_POLICY: Optional[EmbeddedPolicy] = None
_INFERENCE_READY = False


def _load_embedded_policy(action_dim: int) -> Tuple[Optional[EmbeddedPolicy], bool]:
    global _INFERENCE_POLICY, _INFERENCE_READY
    if _INFERENCE_POLICY is not None:
        return _INFERENCE_POLICY, _INFERENCE_READY
    if EMBEDDED_STATE_DICT is None:
        return None, False
    literal = {k: np.asarray(v, dtype=np.float32) for k, v in EMBEDDED_STATE_DICT.items()}
    policy = EmbeddedPolicy(literal)
    if policy.adv_b.shape[0] != action_dim:
        return None, False
    _INFERENCE_POLICY = policy
    _INFERENCE_READY = True
    return policy, True


def my_agent(observation, configuration):
    board = _extract_board(observation)
    playable = find_playable_columns(board, configuration)
    if not playable:
        return 0

    policy, ready = _load_embedded_policy(_get_cfg_value(configuration, "columns", 7))
    if not ready or policy is None:
        return fallback_move(playable, configuration)

    state = state_from_board(board, configuration)
    return policy.act(state, playable)


def export_state_dict_literal(model: DuelingQNetwork) -> str:
    if not HAS_TORCH:
        raise ImportError("PyTorch 없이 state_dict를 추출할 수 없습니다.")
    lines = ["EMBEDDED_STATE_DICT = {"]
    for name, tensor in model.state_dict().items():
        lines.append(f"    {name!r}: {tensor.detach().cpu().tolist()},")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    trained = train_dueling_dqn_agent(episodes=1000)
    literal = export_state_dict_literal(trained)
    if EMBEDDED_STATE_OUTPUT is not None:
        EMBEDDED_STATE_OUTPUT.write_text(literal, encoding="utf-8")
        print(f"[INFO] wrote literal to {EMBEDDED_STATE_OUTPUT}")
    print(literal)
