from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: float):
        ctx.save_for_backward(x)
        ctx.gamma = gamma
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        gamma = ctx.gamma
        grad = gamma / (1.0 + x.abs()).pow(2)
        return grad_output * grad, None


def spike_fn(x: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    return SurrogateSpike.apply(x, gamma)


@dataclass
class RSNNConfig:
    # =========================
    # 1) Encoder / decoder shape
    # =========================
    input_size: int = 256
    hidden_size: int = 512
    output_size: int = 11

    in_channels: int = 2
    encoder_base_channels: int = 32
    encoder_dropout: float = 0.3
    frame_height: int = 32
    frame_width: int = 32

    output_bias: bool = True
    readout_mode: str = "rate"  # rate | last_spike | last_membrane

    # =========================
    # 2) Core neuron dynamics
    # =========================
    static_repeat_steps: int = 5
    beta: float = 0.9
    threshold: float = 1.0
    reset_mode: str = "subtract"  # subtract | zero
    surrogate_gamma: float = 0.5
    detach_reset: bool = True
    use_input_drive_each_step: bool = True

    use_layernorm: bool = False
    norm_location: str = "none"  # none | input | pre_spike
    clamp_hidden: Optional[float] = None
    spectral_radius: Optional[float] = 0.9
    weight_clip_value: Optional[float] = None

    use_residual: bool = False
    residual_alpha: float = 0.1
    use_noise: bool = False
    noise_std: float = 0.01

    # =========================
    # 3) Connectivity / sparsity
    # =========================
    use_sparse_connectivity: bool = True
    sparsity: float = 0.05
    allow_self_connections: bool = False

    use_learnable_mask: bool = True
    mask_mode: str = "ste_topk"  # ste_topk | soft | contextual_topk
    mask_temperature: float = 1.0
    mask_ste_scale: float = 10.0
    mask_score_init_std: float = 0.01
    gate_l1_lambda: float = 0.0
    selector_hidden_dim: int = 128
    selector_source: str = "input_spike"  # input | spike | membrane | input_spike | input_membrane | spike_membrane | input_spike_membrane
    selector_topk: Optional[int] = None
    selector_use_base_scores: bool = True

    use_activity_tracking: bool = True
    activity_decay: float = 0.95
    target_activity: float = 0.05

    # =========================
    # 4) Plasticity / adaptation
    # =========================
    enable_plasticity: bool = False

    use_hebbian: bool = False
    hebbian_lr: float = 1e-4
    hebbian_decay: float = 0.0
    hebbian_center: bool = False
    hebbian_post_type: str = "spike"  # spike | membrane
    hebbian_normalize_by_length: bool = True

    use_homeostasis: bool = False
    homeostasis_lr: float = 1e-4
    homeostasis_mode: str = "bias"  # bias | incoming | outgoing

    use_rewiring: bool = False
    prune_rate: float = 0.01
    growth_rate: float = 0.01
    keep_constant_density: bool = True
    new_weight_scale: float = 0.01

    # =========================
    # 5) Sign constraints
    # =========================
    use_ei: bool = False
    excitatory_ratio: float = 0.8

    def __post_init__(self):
        if not self.use_sparse_connectivity:
            self.use_learnable_mask = False
        if self.frame_height % 8 != 0 or self.frame_width % 8 != 0:
            raise ValueError("frame_height and frame_width must both be divisible by 8.")

    @property
    def encoder_feature_dim(self) -> int:
        return self.encoder_base_channels * 4 * (self.frame_height // 8) * (self.frame_width // 8)

    @property
    def sparsity_enabled(self) -> bool:
        return self.use_sparse_connectivity

    @property
    def plasticity_enabled(self) -> bool:
        return self.enable_plasticity and (self.use_hebbian or self.use_homeostasis or self.use_rewiring)


class RSNNCore(nn.Module):
    def __init__(self, cfg: RSNNConfig):
        super().__init__()
        self.cfg = cfg

        H = cfg.hidden_size
        I = cfg.input_size
        O = cfg.output_size

        self.W_in = nn.Parameter(torch.empty(I, H))
        self.W_rec = nn.Parameter(torch.empty(H, H))
        self.W_out = nn.Parameter(torch.empty(H, O))
        self.h_bias = nn.Parameter(torch.zeros(H))

        if cfg.output_bias:
            self.b_out = nn.Parameter(torch.zeros(O))
        else:
            self.register_parameter("b_out", None)

        valid_edges = torch.ones(H, H)
        if not cfg.allow_self_connections:
            valid_edges.fill_diagonal_(0.0)
        self.register_buffer("valid_edges", valid_edges)

        if cfg.sparsity_enabled:
            base_mask = (torch.rand(H, H) < cfg.sparsity).float() * valid_edges
        else:
            base_mask = valid_edges.clone()
        self.register_buffer("mask", base_mask)

        if cfg.sparsity_enabled and cfg.use_learnable_mask:
            scores = torch.randn(H, H) * cfg.mask_score_init_std
            centered = base_mask * 2.0 - 1.0
            scores = scores + centered
            scores = scores * valid_edges + (-1e4) * (1.0 - valid_edges)
            self.mask_scores = nn.Parameter(scores)
        else:
            self.register_parameter("mask_scores", None)

        if cfg.sparsity_enabled and cfg.mask_mode == "contextual_topk":
            source_dim_map = {
                "input": H,
                "spike": H,
                "membrane": H,
                "input_spike": H * 2,
                "input_membrane": H * 2,
                "spike_membrane": H * 2,
                "input_spike_membrane": H * 3,
            }
            if cfg.selector_source not in source_dim_map:
                raise ValueError(f"Unsupported selector_source: {cfg.selector_source}")
            selector_in_dim = source_dim_map[cfg.selector_source]
            self.selector_backbone = nn.Sequential(
                nn.Linear(selector_in_dim, cfg.selector_hidden_dim),
                nn.GELU(),
            )
            self.selector_row = nn.Linear(cfg.selector_hidden_dim, H)
            self.selector_col = nn.Linear(cfg.selector_hidden_dim, H)
        else:
            self.selector_backbone = None
            self.selector_row = None
            self.selector_col = None

        self.register_buffer("activity", torch.zeros(H))

        if cfg.use_ei:
            n_exc = int(round(H * cfg.excitatory_ratio))
            ei_sign = torch.ones(H)
            ei_sign[n_exc:] = -1.0
        else:
            ei_sign = torch.ones(H)
        self.register_buffer("ei_sign", ei_sign)

        self.last_metrics: Dict[str, float] = {}
        self.norm = nn.LayerNorm(H) if cfg.use_layernorm else nn.Identity()
        self.input_norm = nn.LayerNorm(H) if cfg.use_layernorm and cfg.norm_location == "input" else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)
        nn.init.orthogonal_(self.W_rec)
        with torch.no_grad():
            if not (self.cfg.sparsity_enabled and self.cfg.use_learnable_mask):
                self.W_rec.mul_(self.mask)
            self._apply_ei_constraint_()
            self._enforce_stability_()

    def _num_active_connections(self) -> int:
        total_valid = int(self.valid_edges.sum().item())
        if not self.cfg.sparsity_enabled:
            return total_valid
        k = int(round(self.cfg.sparsity * total_valid))
        return max(1, min(total_valid, k)) if total_valid > 0 else 0

    def _selector_context(
        self,
        input_current: Optional[torch.Tensor] = None,
        spike_state: Optional[torch.Tensor] = None,
        membrane_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source = self.cfg.selector_source
        parts = []
        if "input" in source:
            if input_current is None:
                raise ValueError("input_current is required by selector_source.")
            parts.append(input_current)
        if "spike" in source:
            if spike_state is None:
                raise ValueError("spike_state is required by selector_source.")
            parts.append(spike_state)
        if "membrane" in source:
            if membrane_state is None:
                raise ValueError("membrane_state is required by selector_source.")
            parts.append(membrane_state)
        if not parts:
            raise ValueError(f"Unsupported selector_source: {source}")
        return torch.cat(parts, dim=-1)

    def _base_gate_scores(self) -> torch.Tensor:
        if self.mask_scores is not None:
            return self.mask_scores.masked_fill(self.valid_edges == 0, -1e4)
        if self.cfg.sparsity_enabled:
            return (self.mask * 2.0 - 1.0).masked_fill(self.valid_edges == 0, -1e4)
        return self.valid_edges.masked_fill(self.valid_edges == 0, -1e4)

    def _hard_topk_gate_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        k = self.cfg.selector_topk if self.cfg.selector_topk is not None else self._num_active_connections()
        if scores.dim() == 2:
            flat_scores = scores.view(-1)
            valid_flat = self.valid_edges.view(-1) > 0
            hard_flat = torch.zeros_like(flat_scores)
            if k > 0:
                valid_scores = flat_scores[valid_flat]
                topk_idx = torch.topk(valid_scores, k=k, sorted=False).indices
                valid_positions = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
                hard_flat[valid_positions[topk_idx]] = 1.0
            return hard_flat.view_as(scores) * self.valid_edges

        if scores.dim() != 3:
            raise ValueError("scores must be [H, H] or [B, H, H].")

        B = scores.size(0)
        flat_scores = scores.view(B, -1)
        valid_flat = self.valid_edges.view(-1).bool().unsqueeze(0).expand(B, -1)
        masked_scores = flat_scores.masked_fill(~valid_flat, -1e4)
        hard_flat = torch.zeros_like(masked_scores)
        if k > 0:
            topk_idx = torch.topk(masked_scores, k=k, dim=1, sorted=False).indices
            hard_flat.scatter_(1, topk_idx, 1.0)
        return hard_flat.view_as(scores) * self.valid_edges.unsqueeze(0)

        total_valid = int(self.valid_edges.sum().item())
        if not self.cfg.sparsity_enabled:
            return total_valid
        k = int(round(self.cfg.sparsity * total_valid))
        return max(1, min(total_valid, k)) if total_valid > 0 else 0

    def _compute_gate(
        self,
        training: bool,
        input_current: Optional[torch.Tensor] = None,
        spike_state: Optional[torch.Tensor] = None,
        membrane_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.cfg.sparsity_enabled:
            return self.valid_edges

        if self.cfg.mask_mode == "contextual_topk":
            if self.selector_backbone is None or self.selector_row is None or self.selector_col is None:
                raise RuntimeError("contextual_topk requires selector modules to be initialized.")
            context = self._selector_context(
                input_current=input_current,
                spike_state=spike_state,
                membrane_state=membrane_state,
            )
            hidden = self.selector_backbone(context)
            row_scores = self.selector_row(hidden)
            col_scores = self.selector_col(hidden)
            scores = row_scores.unsqueeze(2) + col_scores.unsqueeze(1)
            if self.cfg.selector_use_base_scores:
                scores = scores + self._base_gate_scores().unsqueeze(0)
            scores = scores.masked_fill(self.valid_edges.unsqueeze(0) == 0, -1e4)

            hard_gate = self._hard_topk_gate_from_scores(scores)
            if training:
                soft_gate = torch.sigmoid(scores * self.cfg.mask_ste_scale / max(self.cfg.mask_temperature, 1e-6))
                gate = hard_gate + soft_gate - soft_gate.detach()
            else:
                gate = hard_gate
            return gate * self.valid_edges.unsqueeze(0)

        if self.cfg.use_learnable_mask and self.mask_scores is not None:
            scores = self.mask_scores.masked_fill(self.valid_edges == 0, -1e4)
            if self.cfg.mask_mode == "soft":
                gate = torch.sigmoid(scores / max(self.cfg.mask_temperature, 1e-6))
                return gate * self.valid_edges

            if self.cfg.mask_mode != "ste_topk":
                raise ValueError(f"Unsupported mask_mode: {self.cfg.mask_mode}")

            hard_gate = self._hard_topk_gate_from_scores(scores)
            if training:
                soft_gate = torch.sigmoid(scores * self.cfg.mask_ste_scale / max(self.cfg.mask_temperature, 1e-6))
                gate = hard_gate + soft_gate - soft_gate.detach()
            else:
                gate = hard_gate
            return gate * self.valid_edges

        return self.mask * self.valid_edges

    def effective_W_rec(
        self,
        training: Optional[bool] = None,
        gate_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if training is None:
            training = self.training
        gate = self._compute_gate(training=training) if gate_override is None else gate_override
        if gate.dim() == 3:
            W = self.W_rec.unsqueeze(0) * gate
            if self.cfg.use_ei:
                W = torch.abs(W) * self.ei_sign.view(1, -1, 1)
            return W
        W = self.W_rec * gate
        if self.cfg.use_ei:
            W = torch.abs(W) * self.ei_sign.unsqueeze(1)
        return W

    @torch.no_grad()
    def _apply_ei_constraint_(self):
        if self.cfg.use_ei:
            self.W_rec.copy_(torch.abs(self.W_rec) * self.ei_sign.unsqueeze(1))
        if self.cfg.sparsity_enabled and not self.cfg.use_learnable_mask:
            self.W_rec.mul_(self.mask)
        self.W_rec.mul_(self.valid_edges)
        if self.mask_scores is not None:
            self.mask_scores.data.mul_(self.valid_edges).add_((-1e4) * (1.0 - self.valid_edges))

    @torch.no_grad()
    def _enforce_stability_(self):
        self._apply_ei_constraint_()
        if self.cfg.weight_clip_value is not None:
            self.W_rec.clamp_(-self.cfg.weight_clip_value, self.cfg.weight_clip_value)
        if self.cfg.spectral_radius is not None:
            if self.cfg.mask_mode == "contextual_topk":
                proxy_gate = self._hard_topk_gate_from_scores(self._base_gate_scores())
                W_eff = self.effective_W_rec(training=False, gate_override=proxy_gate)
            else:
                W_eff = self.effective_W_rec(training=False)
            norm = torch.linalg.matrix_norm(W_eff, ord=2)
            target = float(self.cfg.spectral_radius)
            if torch.isfinite(norm) and norm > target and norm > 0:
                self.W_rec.mul_(target / (norm + 1e-8))
        if self.cfg.sparsity_enabled and not self.cfg.use_learnable_mask:
            self.W_rec.mul_(self.mask)
        self.W_rec.mul_(self.valid_edges)

    @torch.no_grad()
    def stabilize_(self):
        self._enforce_stability_()

    @torch.no_grad()
    def hard_mask_(self):
        if self.cfg.sparsity_enabled and not self.cfg.use_learnable_mask:
            self.W_rec.mul_(self.mask)
        self.W_rec.mul_(self.valid_edges)

    def gate_regularization_loss(self) -> torch.Tensor:
        if not (self.cfg.sparsity_enabled and self.cfg.use_learnable_mask and self.mask_scores is not None):
            return self.W_rec.new_zeros(())
        gate_soft = torch.sigmoid(self.mask_scores / max(self.cfg.mask_temperature, 1e-6)) * self.valid_edges
        return gate_soft.mean() * self.cfg.gate_l1_lambda

    def _lif_step(
        self,
        input_current: torch.Tensor,
        recurrent_current: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        input_current = self.input_norm(input_current)
        total = input_current + recurrent_current + self.h_bias
        if cfg.use_residual:
            total = total + cfg.residual_alpha * v
        if cfg.use_noise and self.training:
            total = total + torch.randn_like(total) * cfg.noise_std

        v = cfg.beta * v + total
        if cfg.use_layernorm and cfg.norm_location == "pre_spike":
            v = self.norm(v)
        if cfg.clamp_hidden is not None:
            v = torch.clamp(v, -cfg.clamp_hidden, cfg.clamp_hidden)

        s = spike_fn(v - cfg.threshold, cfg.surrogate_gamma)
        s_reset = s.detach() if cfg.detach_reset else s

        if cfg.reset_mode == "subtract":
            v = v - s_reset * cfg.threshold
        elif cfg.reset_mode == "zero":
            v = v * (1.0 - s_reset)
        else:
            raise ValueError(f"Unsupported reset_mode: {cfg.reset_mode}")
        return v, s

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return steps < lengths.unsqueeze(1)

    @torch.no_grad()
    def _update_activity_trace_(self, spike_tensor: torch.Tensor, valid_mask: torch.Tensor):
        if not self.cfg.use_activity_tracking or spike_tensor.numel() == 0:
            return
        valid = valid_mask.to(spike_tensor.dtype).unsqueeze(-1)
        denom = valid.sum().clamp_min(1.0)
        batch_activity = (spike_tensor * valid).sum(dim=(0, 1)) / denom
        decay = self.cfg.activity_decay
        self.activity.mul_(decay).add_((1.0 - decay) * batch_activity)

    @torch.no_grad()
    def _record_metrics_(
        self,
        v: torch.Tensor,
        s: torch.Tensor,
        out: torch.Tensor,
        gate_override: Optional[torch.Tensor] = None,
    ):
        W_eff = self.effective_W_rec(training=False, gate_override=gate_override)
        gate = self._compute_gate(training=False) if gate_override is None else gate_override
        if gate.dim() == 3:
            active_conn = gate.sum(dim=(1, 2)).float().mean().item()
        else:
            active_conn = gate.sum().item() if self.cfg.sparsity_enabled else self.valid_edges.sum().item()
        density = float(active_conn / self.valid_edges.sum().clamp_min(1.0).item())
        self.last_metrics = {
            "activity_mean": float(self.activity.mean().item()),
            "activity_std": float(self.activity.std().item()),
            "membrane_abs_mean": float(v.abs().mean().item()),
            "spike_rate": float(s.float().mean().item()),
            "out_abs_mean": float(out.abs().mean().item()),
            "rec_weight_abs_mean": float(W_eff.abs().mean().item()),
            "density": density,
        }
        if self.cfg.sparsity_enabled and self.cfg.use_learnable_mask and self.mask_scores is not None:
            gate_soft = torch.sigmoid(self.mask_scores / max(self.cfg.mask_temperature, 1e-6)) * self.valid_edges
            self.last_metrics.update(
                {
                    "gate_soft_mean": float(gate_soft.sum().item() / self.valid_edges.sum().clamp_min(1.0).item()),
                    "gate_score_mean": float((self.mask_scores * self.valid_edges).sum().item() / self.valid_edges.sum().clamp_min(1.0).item()),
                }
            )

    def forward_features(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        cfg = self.cfg

        if x.dim() == 2:
            B = x.size(0)
            input_seq = [x @ self.W_in for _ in range(cfg.static_repeat_steps)]
            num_steps = len(input_seq)
            if lengths is None:
                lengths = torch.full((B,), num_steps, device=x.device, dtype=torch.long)
        elif x.dim() == 3:
            B, T, I = x.shape
            if I != cfg.input_size:
                raise ValueError(f"Expected input_size={cfg.input_size}, got {I}")
            input_seq = [x[:, t, :] @ self.W_in for t in range(T)]
            num_steps = T
            if lengths is None:
                lengths = torch.full((B,), T, device=x.device, dtype=torch.long)
        else:
            raise ValueError("x must be [B, I] or [B, T, I]")

        if mask is None:
            valid_mask = self._lengths_to_mask(lengths.to(x.device), num_steps)
        else:
            valid_mask = mask.to(dtype=torch.bool, device=x.device)
            lengths = valid_mask.sum(dim=1).long()

        H = cfg.hidden_size
        v = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        s = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        spike_sum = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        last_valid_spike = torch.zeros_like(spike_sum)
        last_valid_membrane = torch.zeros_like(spike_sum)

        spike_states: List[torch.Tensor] = []
        membrane_states: List[torch.Tensor] = []

        W_rec_eff = None if cfg.mask_mode == "contextual_topk" else self.effective_W_rec(training=self.training)
        zero_inp = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        last_gate = None

        for t in range(num_steps):
            valid_t = valid_mask[:, t].to(x.dtype).unsqueeze(1)
            inp = input_seq[t] if cfg.use_input_drive_each_step else (input_seq[t] if t == 0 else zero_inp)

            if cfg.mask_mode == "contextual_topk":
                gate_t = self._compute_gate(
                    training=self.training,
                    input_current=inp,
                    spike_state=s,
                    membrane_state=v,
                )
                W_rec_eff_t = self.effective_W_rec(training=self.training, gate_override=gate_t)
                rec = torch.bmm(s.unsqueeze(1), W_rec_eff_t).squeeze(1)
                last_gate = gate_t.detach()
            else:
                rec = s @ W_rec_eff

            v_new, s_new = self._lif_step(inp, rec, v)
            v = valid_t * v_new + (1.0 - valid_t) * v
            s = valid_t * s_new
            spike_states.append(s)
            membrane_states.append(v)
            spike_sum = spike_sum + s
            last_valid_spike = valid_t * s + (1.0 - valid_t) * last_valid_spike
            last_valid_membrane = valid_t * v + (1.0 - valid_t) * last_valid_membrane

        denom = lengths.clamp_min(1).to(x.dtype).unsqueeze(1)
        if cfg.readout_mode == "rate":
            readout_h = spike_sum / denom
        elif cfg.readout_mode == "last_spike":
            readout_h = last_valid_spike
        elif cfg.readout_mode == "last_membrane":
            readout_h = last_valid_membrane
        else:
            raise ValueError(f"Unsupported readout_mode: {cfg.readout_mode}")

        spike_tensor = torch.stack(spike_states, dim=1)
        membrane_tensor = torch.stack(membrane_states, dim=1)
        with torch.no_grad():
            self._update_activity_trace_(spike_tensor, valid_mask)
        if return_state:
            state = {
                "spikes": spike_tensor,
                "membranes": membrane_tensor,
                "final_spike": last_valid_spike,
                "final_membrane": last_valid_membrane,
                "readout_h": readout_h,
                "activity": self.activity.detach().clone(),
                "metrics": dict(self.last_metrics),
                "lengths": lengths.detach().clone(),
                "valid_mask": valid_mask.detach().clone(),
            }
            if self.cfg.sparsity_enabled:
                if cfg.mask_mode == "contextual_topk" and last_gate is not None:
                    state["rec_gate"] = last_gate.clone()
                else:
                    state["rec_gate"] = self._compute_gate(training=False).detach().clone()
            return readout_h, state
        return readout_h

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        features = self.forward_features(x, lengths=lengths, mask=mask, return_state=return_state)
        if return_state:
            readout_h, state = features
            out = readout_h @ self.W_out
            if self.b_out is not None:
                out = out + self.b_out
            with torch.no_grad():
                self._record_metrics_(
                    state["final_membrane"],
                    state["final_spike"],
                    out,
                    gate_override=state.get("rec_gate"),
                )
            return out, state

        readout_h = features
        out = readout_h @ self.W_out
        if self.b_out is not None:
            out = out + self.b_out
        with torch.no_grad():
            dummy = torch.zeros_like(readout_h)
            self._record_metrics_(dummy, dummy, out)
        return out

    @torch.no_grad()
    def hebbian_update_from_states(
        self,
        spike_states: torch.Tensor,
        membrane_states: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        lr: Optional[float] = None,
    ):
        if not (self.cfg.plasticity_enabled and self.cfg.use_hebbian):
            return
        if spike_states is None or spike_states.dim() != 3 or spike_states.size(1) < 2:
            return
        if lr is None:
            lr = self.cfg.hebbian_lr

        B, T, _ = spike_states.shape
        if valid_mask is None:
            if lengths is None:
                valid_mask = torch.ones(B, T, dtype=torch.bool, device=spike_states.device)
            else:
                valid_mask = self._lengths_to_mask(lengths.to(spike_states.device), T)

        if self.cfg.hebbian_post_type == "membrane":
            if membrane_states is None:
                raise ValueError("membrane_states is required when hebbian_post_type='membrane'.")
            post_source = membrane_states
        else:
            post_source = spike_states

        dw_acc = torch.zeros_like(self.W_rec)
        total_pairs = 0.0

        for t in range(T - 1):
            pair_mask = (valid_mask[:, t] & valid_mask[:, t + 1]).to(spike_states.dtype).unsqueeze(1)
            if pair_mask.sum() <= 0:
                continue
            pre = spike_states[:, t, :] * pair_mask
            post = post_source[:, t + 1, :] * pair_mask
            if self.cfg.hebbian_center:
                pre = pre - pre.mean(dim=0, keepdim=True)
                post = post - post.mean(dim=0, keepdim=True)
            dw_acc += pre.t() @ post
            total_pairs += float(pair_mask.sum().item())

        if total_pairs <= 0:
            return

        dw_acc = dw_acc / total_pairs
        if self.cfg.hebbian_normalize_by_length:
            dw_acc = dw_acc / max(T - 1, 1)

        if self.cfg.hebbian_decay > 0.0:
            self.W_rec.mul_(1.0 - self.cfg.hebbian_decay)
        self.W_rec.add_(lr * dw_acc)
        self.W_rec.mul_(self.valid_edges)
        if self.cfg.sparsity_enabled and not self.cfg.use_learnable_mask:
            self.W_rec.mul_(self.mask)
        self._enforce_stability_()

    @torch.no_grad()
    def homeostasis(self, lr: Optional[float] = None):
        if not (self.cfg.plasticity_enabled and self.cfg.use_homeostasis):
            return
        if lr is None:
            lr = self.cfg.homeostasis_lr
        delta = self.cfg.target_activity - self.activity
        if self.cfg.homeostasis_mode == "bias":
            self.h_bias.add_(lr * delta)
        elif self.cfg.homeostasis_mode == "incoming":
            self.W_rec.add_(lr * delta.unsqueeze(0))
        elif self.cfg.homeostasis_mode == "outgoing":
            self.W_rec.add_(lr * delta.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported homeostasis_mode: {self.cfg.homeostasis_mode}")
        self.W_rec.mul_(self.valid_edges)
        if self.cfg.sparsity_enabled and not self.cfg.use_learnable_mask:
            self.W_rec.mul_(self.mask)
        self._enforce_stability_()

    @torch.no_grad()
    def rewiring(self):
        if not (self.cfg.plasticity_enabled and self.cfg.use_rewiring and self.cfg.sparsity_enabled):
            return

        if self.cfg.use_learnable_mask and self.mask_scores is not None:
            activity_pair = torch.outer(self.activity, self.activity)
            activity_pair = activity_pair * self.valid_edges
            keep_bias = activity_pair - activity_pair.mean()
            self.mask_scores.add_(self.cfg.growth_rate * keep_bias)
            self.mask_scores.data.masked_fill_(self.valid_edges == 0, -1e4)
            return

        active = self.mask.bool()
        if active.sum() == 0:
            return

        W_eff = self.effective_W_rec(training=False)
        active_weights = W_eff.abs()[active]
        if active_weights.numel() == 0:
            return

        prune_ratio = float(max(0.0, min(1.0, self.cfg.prune_rate)))
        threshold = torch.quantile(active_weights, prune_ratio)
        prune = active & (W_eff.abs() < threshold)
        if not self.cfg.allow_self_connections:
            diag = torch.eye(self.cfg.hidden_size, dtype=torch.bool, device=self.mask.device)
            prune = prune & (~diag)
        num_pruned = int(prune.sum().item())
        if num_pruned == 0:
            return

        self.mask[prune] = 0.0
        self.W_rec[prune] = 0.0

        grow_candidates = self.mask == 0
        if not self.cfg.allow_self_connections:
            grow_candidates.fill_diagonal_(False)
        idx = torch.nonzero(grow_candidates, as_tuple=False)
        if idx.numel() == 0:
            return

        if self.cfg.keep_constant_density:
            num_new = min(num_pruned, idx.size(0))
        else:
            num_new = min(int(self.cfg.growth_rate * (self.cfg.hidden_size ** 2)), idx.size(0))
        if num_new <= 0:
            return

        scores = self.activity[idx[:, 0]] * self.activity[idx[:, 1]]
        perm = torch.topk(scores, k=num_new, sorted=False).indices
        chosen = idx[perm]
        self.mask[chosen[:, 0], chosen[:, 1]] = 1.0

        new_values = torch.randn(num_new, device=self.W_rec.device, dtype=self.W_rec.dtype) * self.cfg.new_weight_scale
        if self.cfg.use_ei:
            row_sign = self.ei_sign[chosen[:, 0]]
            new_values = new_values.abs() * row_sign
        self.W_rec[chosen[:, 0], chosen[:, 1]] = new_values
        self._enforce_stability_()

    @torch.no_grad()
    def apply_plasticity(
        self,
        state: Optional[Dict[str, torch.Tensor]] = None,
        do_hebbian: Optional[bool] = None,
        do_homeostasis: Optional[bool] = None,
        do_rewiring: Optional[bool] = None,
    ):
        if not self.cfg.plasticity_enabled:
            return

        if do_hebbian is None:
            do_hebbian = self.cfg.use_hebbian
        if do_homeostasis is None:
            do_homeostasis = self.cfg.use_homeostasis
        if do_rewiring is None:
            do_rewiring = self.cfg.use_rewiring

        if do_hebbian and state is not None:
            self.hebbian_update_from_states(
                spike_states=state.get("spikes"),
                membrane_states=state.get("membranes"),
                lengths=state.get("lengths"),
                valid_mask=state.get("valid_mask"),
            )
        if do_homeostasis:
            self.homeostasis()
        if do_rewiring:
            self.rewiring()

    @torch.no_grad()
    def apply_mechanisms(self, state: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        self.apply_plasticity(state=state, **kwargs)


class ConvFrameEncoder(nn.Module):
    def __init__(self, cfg: RSNNConfig):
        super().__init__()
        base = cfg.encoder_base_channels
        self.net = nn.Sequential(
            nn.Conv2d(cfg.in_channels, base, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.proj = nn.Linear(cfg.encoder_feature_dim, cfg.input_size)
        self.dropout = nn.Dropout(cfg.encoder_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.dropout(self.proj(h))


class RSNNDecoder(nn.Module):
    def __init__(self, core: RSNNCore):
        super().__init__()
        self.core = core

    def forward(self, readout_h: torch.Tensor) -> torch.Tensor:
        out = readout_h @ self.core.W_out
        if self.core.b_out is not None:
            out = out + self.core.b_out
        return out


class RSNN(nn.Module):
    def __init__(self, cfg: RSNNConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvFrameEncoder(cfg)
        self.core = RSNNCore(cfg)
        self.decoder = RSNNDecoder(self.core)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5:
            raise ValueError("frames must be [B, T, C, H, W]")
        B, T, C, H, W = frames.shape
        if C != self.cfg.in_channels:
            raise ValueError(f"Expected in_channels={self.cfg.in_channels}, got {C}")
        if H != self.cfg.frame_height or W != self.cfg.frame_width:
            raise ValueError(
                f"Expected frame size ({self.cfg.frame_height}, {self.cfg.frame_width}), got ({H}, {W})"
            )
        feat = self.encoder(frames.reshape(B * T, C, H, W))
        return feat.reshape(B, T, self.cfg.input_size)

    def forward(
        self,
        frames: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        feat = self.encode_frames(frames)
        core_out = self.core.forward_features(feat, lengths=lengths, mask=mask, return_state=return_state)

        if return_state:
            readout_h, state = core_out
            logits = self.decoder(readout_h)
            with torch.no_grad():
                self.core._record_metrics_(
                    state["final_membrane"],
                    state["final_spike"],
                    logits,
                    gate_override=state.get("rec_gate"),
                )
            return logits, state

        readout_h = core_out
        logits = self.decoder(readout_h)
        with torch.no_grad():
            self.core._record_metrics_(torch.zeros_like(readout_h), torch.zeros_like(readout_h), logits)
        return logits

    @torch.no_grad()
    def apply_plasticity(self, state: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        self.core.apply_plasticity(state=state, **kwargs)

    @torch.no_grad()
    def apply_mechanisms(self, state: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        self.core.apply_plasticity(state=state, **kwargs)

    def auxiliary_loss(self) -> torch.Tensor:
        return self.core.gate_regularization_loss()

    def extra_repr(self) -> str:
        return (
            f"input={self.cfg.input_size}, hidden={self.cfg.hidden_size}, output={self.cfg.output_size}, "
            f"readout={self.cfg.readout_mode}, sparse={self.cfg.sparsity_enabled}, plasticity={self.cfg.plasticity_enabled}"
        )
