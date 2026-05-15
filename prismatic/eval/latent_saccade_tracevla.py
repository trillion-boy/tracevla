"""
latent_saccade_tracevla.py

LatentSaccade ported from UniVLA (VQ-VAE discrete tokens) to TraceVLA
(continuous ViT patch embeddings).

Architecture differences handled here:
  UniVLA : Image → VQ-VAE → token IDs → embed_tokens → hook → LLM
  TraceVLA: Image → SigLIP+DINOv2 ViT → projector → hook → LLaMA-2

Hook location:
  Register a forward hook on `vla.projector` (the MLP adapter).
  The hook fires once per predict_action() call (only during prefill —
  cached generation steps do NOT call the projector).
  Scales the first image's patch embeddings by the spatial weight map.
  The second image (trace overlay) is left unchanged.

Token layout for TraceVLA (2 images, 224px, patch_size=14 → 16×16=256):
  [0   : 256]  original observation patches   ← APPLY WEIGHT MAP HERE
  [256]        separator token                ← untouched
  [257 : 513]  trace-overlay patches          ← untouched (temporal context)

Weight zones:
  fovea      (DINO bbox of task-relevant object) → fovea_weight  (default 1.0)
  secondary  (source object, PLACE phase only)   → place_src_weight (default 0.5)
  background (all remaining patches)             → bg_weight     (default 0.2)

SaccadeStateMachine:
  GRASP phase → fovea on source object
  PLACE phase → fovea on destination, secondary on source held in hand
  Transition  → gripper closes for ≥ min_grasp_steps steps
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection
from transformers import AutoProcessor as _HFAutoProcessor

from .tracevla_simpler import TraceVLAInference, resize_image


# ── tiny helpers that TraceVLAInference assumes exist ─────────────────────────

def _unwrap_ids(ids, num_envs: int) -> List[int]:
    if ids is None:
        return list(range(num_envs))
    return list(ids)


def _get_batch(dicts: List[Dict], key: str) -> List[Any]:
    return [d[key] for d in dicts]


# ══════════════════════════════════════════════════════════════════════════════
# 1. GroundingDINOWrapper
#    Copied from latent_saccade_standalone.py — lazy-loads HF grounding-dino.
# ══════════════════════════════════════════════════════════════════════════════

class GroundingDINOWrapper:
    """HF Grounding DINO wrapper.  Returns bbox (x1,y1,x2,y2) in pixel coords."""

    _VERB_PATTERNS = [
        r"(?:pick up|grasp|grab|lift|take)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"(?:put|place|move|transfer)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:in|on|into|onto|to)",
        r"(?:push|slide|pull)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"(?:open|close)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"(?:stack)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
    ]

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        print(f"[DINO] Loading {self.model_name} …")
        self._processor = _HFAutoProcessor.from_pretrained(self.model_name)
        self._model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(self.model_name)
            .to(self.device).eval()
        )
        print("[DINO] Loaded.")

    @staticmethod
    def extract_target_noun(instruction: str) -> str:
        instr = instruction.lower().strip()
        for pat in GroundingDINOWrapper._VERB_PATTERNS:
            m = re.search(pat, instr)
            if m:
                return m.group(1).strip()
        return instr

    @staticmethod
    def extract_source_dest_nouns(instruction: str) -> Tuple[str, Optional[str]]:
        instr = instruction.lower().strip()
        pat = (
            r"(?:stack|put|place|move|transfer)\s+(?:the\s+)?(.+?)"
            r"\s+(?:in|on|into|onto|to)\s+(?:the\s+)?(.+?)(?:\s*$|\s+and\s)"
        )
        m = re.search(pat, instr)
        if m:
            return m.group(1).strip().rstrip(".,"), m.group(2).strip().rstrip(".,")
        return GroundingDINOWrapper.extract_target_noun(instruction), None

    def detect_bbox(
        self, image_rgb: np.ndarray, text_query: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """Returns (x1, y1, x2, y2) of highest-confidence box, or None."""
        self._load_model()
        pil_img = Image.fromarray(image_rgb)
        query = text_query if text_query.endswith(".") else text_query + "."
        inputs = self._processor(images=pil_img, text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([pil_img.size[::-1]])
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=self.box_threshold,
                target_sizes=target_sizes,
            )[0]
        if len(results["boxes"]) == 0:
            return None
        best_idx = results["scores"].argmax().item()
        score = results["scores"][best_idx].item()
        box = results["boxes"][best_idx].cpu().numpy()
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        print(f"[DINO] '{text_query.rstrip('.')}' score={score:.3f} → [{x1},{y1},{x2},{y2}]")
        return x1, y1, x2, y2


# ══════════════════════════════════════════════════════════════════════════════
# 2. SaccadeStateMachine
#    Copied from latent_saccade_standalone.py
# ══════════════════════════════════════════════════════════════════════════════

class SaccadeStateMachine:
    GRASP = "grasp"
    PLACE = "place"

    def __init__(
        self,
        source_noun: str = "",
        dest_noun: str = "",
        close_thresh: float = 0.5,
        min_grasp_steps: int = 15,
        consecutive_close_required: int = 3,
    ):
        self.source_noun = source_noun
        self.dest_noun = dest_noun
        self.close_thresh = close_thresh
        self.min_grasp_steps = min_grasp_steps
        self.consecutive_close_required = consecutive_close_required
        self._state = self.GRASP
        self._grasp_steps = 0
        self._close_count = 0

    def update(self, gripper_norm: float) -> bool:
        """Return True if a phase transition occurred."""
        transitioned = False
        if self._state == self.GRASP:
            self._grasp_steps += 1
            self._close_count = (
                self._close_count + 1 if gripper_norm >= self.close_thresh else 0
            )
            if (
                self._grasp_steps >= self.min_grasp_steps
                and self._close_count >= self.consecutive_close_required
            ):
                self._state = self.PLACE
                self._grasp_steps = 0
                self._close_count = 0
                transitioned = True
                print(f"[Saccade] grasp → place (gripper={gripper_norm:.2f})")
        else:
            if gripper_norm < self.close_thresh:
                self._state = self.GRASP
                self._grasp_steps = 0
                self._close_count = 0
                transitioned = True
                print(f"[Saccade] place → grasp (gripper={gripper_norm:.2f})")
        return transitioned

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_target(self) -> str:
        if self._state == self.PLACE and self.dest_noun:
            return self.dest_noun
        return self.source_noun

    def reset(self):
        self._state = self.GRASP
        self._grasp_steps = 0
        self._close_count = 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. _bbox_to_token_mask
#    Same logic as standalone but parameterised for ViT patch grid.
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_to_token_mask(
    bbox: Tuple[int, int, int, int],
    H_t: int, W_t: int,
    H_img: int, W_img: int,
    margin: int = 2,
) -> torch.Tensor:
    """Pixel bbox (x1,y1,x2,y2) → boolean mask on (H_t, W_t) token grid."""
    x1, y1, x2, y2 = bbox
    tx1 = max(0,   int(x1 * W_t / W_img) - margin)
    ty1 = max(0,   int(y1 * H_t / H_img) - margin)
    tx2 = min(W_t, int(x2 * W_t / W_img) + margin)
    ty2 = min(H_t, int(y2 * H_t / H_img) + margin)
    mask = torch.zeros(H_t, W_t, dtype=torch.bool)
    mask[ty1:ty2, tx1:tx2] = True
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# 4. LatentSaccadeTraceVLAInference
# ══════════════════════════════════════════════════════════════════════════════

class LatentSaccadeTraceVLAInference(TraceVLAInference):
    """
    TraceVLA + LatentSaccade.

    Adds a forward hook on `self.vla.projector` that multiplies the
    first image's projected patch embeddings by a spatial weight map
    derived from Grounding DINO detections.

    Everything else (CoTracker trace overlay, sticky gripper, action
    normalisation …) is inherited from TraceVLAInference unchanged.
    """

    def __init__(
        self,
        # ── TraceVLAInference args ─────────────────────────────────────
        model_path: str,
        cotracker_model_path: str,
        dataset_stats_path: str,
        action_scale: float = 1.0,
        n_action_bins: int = 256,
        sample: bool = False,
        temperature: float = 0.0,
        image_aug: bool = False,
        model_dtype: Optional[str] = None,
        device: int = 0,
        # ── LatentSaccade args ────────────────────────────────────────
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        bg_weight: float = 0.2,
        place_src_weight: float = 0.5,
        fovea_weight: float = 1.0,
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        bbox_margin: int = 2,
        min_grasp_steps: int = 15,
        consecutive_close_required: int = 3,
        enable_latent_mask: bool = True,
    ) -> None:
        super().__init__(
            model_path=model_path,
            cotracker_model_path=cotracker_model_path,
            dataset_stats_path=dataset_stats_path,
            action_scale=action_scale,
            n_action_bins=n_action_bins,
            sample=sample,
            temperature=temperature,
            image_aug=image_aug,
            model_dtype=model_dtype,
            device=device,
        )

        self._bg_weight = bg_weight
        self._place_src_weight = place_src_weight
        self._fovea_weight = fovea_weight
        self._dino_cache_steps = dino_cache_steps
        self._bbox_margin = bbox_margin
        self._min_grasp_steps = min_grasp_steps
        self._consecutive_close_required = consecutive_close_required
        self._enable_latent_mask = enable_latent_mask

        self.dino = GroundingDINOWrapper(
            model_name=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=str(self.vla.device),
        )

        self._patch_grid_size = self._detect_patch_grid_size()
        print(f"[LatentSaccade] patch grid = {self._patch_grid_size}×{self._patch_grid_size} "
              f"({self._patch_grid_size**2} tokens per image)")

        # Hook state — set per-env before each predict_action() call
        self._current_weight_map: Optional[torch.Tensor] = None
        self._register_projector_hook()

        # Per-env saccade state — populated in start() / start_episode()
        self.saccades: List[Optional[SaccadeStateMachine]] = []
        self.fovea_cache: List[Optional[Tuple]] = []
        self.secondary_cache: List[Optional[Tuple]] = []
        self.dino_cache_step: List[int] = []

    # ── patch grid auto-detection ─────────────────────────────────────────────

    def _detect_patch_grid_size(self) -> int:
        img_size = self.vla.config.image_sizes[0]          # e.g. 224
        try:
            patch_size = (
                self.vla.vision_backbone.featurizer
                .patch_embed.patch_size[0]
            )
        except Exception:
            patch_size = 14                                 # ViT-L/14 default
        grid = img_size // patch_size
        return grid

    # ── projector hook ────────────────────────────────────────────────────────

    def _register_projector_hook(self):
        N = self._patch_grid_size ** 2  # patches for one image (e.g. 256)

        def _hook(module, inp, output):
            # output: [B, N_total, llm_dim]
            # N_total = 2*N+1 for TraceVLA (two images + separator)
            if not self._enable_latent_mask or self._current_weight_map is None:
                return output
            w = self._current_weight_map.to(dtype=output.dtype, device=output.device)
            out = output.clone()
            # Apply ONLY to first image's tokens; separator + trace image untouched
            out[:, :N, :] = output[:, :N, :] * w.view(1, N, 1)
            return out

        self._hook_handle = self.vla.projector.register_forward_hook(_hook)

    # ── per-env state management ──────────────────────────────────────────────

    def start(self, num_envs: int):
        super().start(num_envs)
        self.saccades = [
            SaccadeStateMachine(
                close_thresh=0.5,
                min_grasp_steps=self._min_grasp_steps,
                consecutive_close_required=self._consecutive_close_required,
            )
            for _ in range(num_envs)
        ]
        self.fovea_cache     = [None] * num_envs
        self.secondary_cache = [None] * num_envs
        self.dino_cache_step = [0]    * num_envs

    def start_episode(
        self,
        obs_dicts: List[Dict],
        info_dicts: List[Dict],
        output_dirs: Optional[List[str]] = None,
        ids: Optional[Union[List[int], np.ndarray]] = None,
    ):
        super().start_episode(obs_dicts, info_dicts, output_dirs, ids)

        ids_list        = _unwrap_ids(ids, self.num_envs)
        task_descs      = _get_batch(info_dicts, "task_description")

        for i, task_desc in enumerate(task_descs):
            idx = ids_list[i]
            src, dst = GroundingDINOWrapper.extract_source_dest_nouns(task_desc)
            self.saccades[idx].source_noun = src
            self.saccades[idx].dest_noun   = dst
            self.saccades[idx].reset()
            self.fovea_cache[idx]     = None
            self.secondary_cache[idx] = None
            self.dino_cache_step[idx] = 0
            print(f"[LatentSaccade] env{idx} instruction → src='{src}'  dst='{dst}'")

    def reset_states_at(self, idx: int, task_description: str):
        super().reset_states_at(idx, task_description)
        if idx < len(self.saccades) and self.saccades[idx] is not None:
            src, dst = GroundingDINOWrapper.extract_source_dest_nouns(task_description)
            self.saccades[idx].source_noun = src
            self.saccades[idx].dest_noun   = dst
            self.saccades[idx].reset()
            self.fovea_cache[idx]     = None
            self.secondary_cache[idx] = None
            self.dino_cache_step[idx] = 0

    # ── DINO bbox with per-env caching ────────────────────────────────────────

    def _get_bboxes_for_env(
        self,
        env_idx: int,
        image_np: np.ndarray,
    ) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        use_cache = (
            self.fovea_cache[env_idx] is not None
            and self.dino_cache_step[env_idx] % self._dino_cache_steps != 0
        )
        if use_cache:
            self.dino_cache_step[env_idx] += 1
            return self.fovea_cache[env_idx], self.secondary_cache[env_idx]

        saccade = self.saccades[env_idx]
        target  = saccade.current_target
        fovea_bbox = self.dino.detect_bbox(image_np, target) if target else None

        secondary_bbox = None
        if saccade.state == SaccadeStateMachine.PLACE and saccade.source_noun:
            secondary_bbox = self.dino.detect_bbox(image_np, saccade.source_noun)

        self.fovea_cache[env_idx]     = fovea_bbox
        self.secondary_cache[env_idx] = secondary_bbox
        self.dino_cache_step[env_idx] += 1
        return fovea_bbox, secondary_bbox

    # ── weight map builder ────────────────────────────────────────────────────

    def _build_weight_map(
        self,
        image_np: np.ndarray,
        fovea_bbox: Optional[Tuple],
        secondary_bbox: Optional[Tuple],
    ) -> Optional[torch.Tensor]:
        """Return flattened weight tensor [H_t*W_t], or None → no masking."""
        if fovea_bbox is None and secondary_bbox is None:
            return None
        H, W   = image_np.shape[:2]
        G      = self._patch_grid_size
        weight = torch.full((G, G), self._bg_weight)

        if secondary_bbox is not None:
            mask = _bbox_to_token_mask(secondary_bbox, G, G, H, W, self._bbox_margin)
            weight[mask] = self._place_src_weight

        if fovea_bbox is not None:
            mask = _bbox_to_token_mask(fovea_bbox, G, G, H, W, self._bbox_margin)
            weight[mask] = self._fovea_weight

        n_fovea = int((weight >= self._fovea_weight).sum())
        n_bg    = int((weight <= self._bg_weight).sum())
        print(f"[LatentSaccade] weight map: fovea={n_fovea}tok  bg={n_bg}tok  "
              f"bbox={fovea_bbox}")
        return weight.reshape(-1)   # [G*G]

    # ── main inference loop ───────────────────────────────────────────────────

    @torch.no_grad()
    def get_action(
        self,
        obs_dicts: List[Dict],
        info_dicts: List[Dict],
        ids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        ids_list         = _unwrap_ids(ids, self.num_envs)
        images           = _get_batch(info_dicts, "image")
        task_descriptions = _get_batch(info_dicts, "task_description")

        assert len(ids_list) == self.num_envs

        # Reset per episode if task description changed
        for i, task_desc in enumerate(task_descriptions):
            if task_desc != self.task_descriptions[i]:
                self.reset_states_at(i, task_desc)

        raw_actions = []
        for i in range(self.num_envs):
            image      = Image.fromarray(images[i])
            image_256  = resize_image(image, (256, 256))
            image_overlaid, has_trace = self.trace_processors[i].process_image(image_256)

            task_description = self.task_descriptions[i]
            image_np = np.array(image_256)

            # ── Saccade: detect bboxes + build weight map ──────────────
            saccade = self.saccades[i]
            print(f"[LatentSaccade] env{i} phase={saccade.state}  "
                  f"target='{saccade.current_target}'")
            fovea_bbox, secondary_bbox = self._get_bboxes_for_env(i, image_np)
            self._current_weight_map = self._build_weight_map(
                image_np, fovea_bbox, secondary_bbox
            )

            # ── Build processor inputs ─────────────────────────────────
            if not has_trace:
                prompt = self.prompt_template.format(task_description=task_description)
                inputs = self.processor(
                    prompt, [image_256, image_256]
                ).to(device=self.vla.device, dtype=torch.bfloat16)
            else:
                prompt = self.prompt_overlaid_template.format(
                    task_description=task_description
                )
                inputs = self.processor(
                    prompt, [image_256, image_overlaid]
                ).to(device=self.vla.device, dtype=torch.bfloat16)

            # ── Run inference (hook fires inside predict_action) ───────
            with torch.inference_mode():
                raw_action = self.vla.predict_action(
                    **inputs,
                    unnorm_key=self.unnorm_keys[i],
                    do_sample=self.sample,
                    temperature=0.7,
                )
            raw_actions.append(raw_action)

            # ── Update saccade state from gripper output ───────────────
            # raw_action[6] = open_gripper (0=open, 1=close, normalised)
            gripper_norm = float(raw_action[6])
            transitioned = saccade.update(gripper_norm)
            if transitioned:
                self.fovea_cache[i]     = None
                self.secondary_cache[i] = None
                self.dino_cache_step[i] = 0

        self._current_weight_map = None  # clear after loop

        # ── Post-process actions (identical to TraceVLAInference) ──────
        from transforms3d.euler import euler2axangle
        actions_list = []
        for i in range(self.num_envs):
            actions = raw_actions[i]
            raw_action = {
                "world_vector":    np.array(actions[:3]),
                "rotation_delta":  np.array(actions[3:6]),
                "open_gripper":    np.array(actions[6:7]),
            }
            action = {}
            action["world_vector"] = raw_action["world_vector"] * self.action_scale
            roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            ax, angle = euler2axangle(roll, pitch, yaw)
            action["rot_axangle"] = ax * angle * self.action_scale

            if self.policy_setups[i] == "widowx_bridge":
                action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

            elif self.policy_setups[i] == "google_robot":
                cur = raw_action["open_gripper"]
                if self.previous_gripper_actions[i] is None:
                    rel = np.array([0])
                else:
                    rel = self.previous_gripper_actions[i] - cur
                self.previous_gripper_actions[i] = cur

                if np.abs(rel) > 0.5 and not self.sticky_action_is_ons[i]:
                    self.sticky_action_is_ons[i]   = True
                    self.sticky_gripper_actions[i] = rel

                if self.sticky_action_is_ons[i]:
                    self.gripper_action_repeats[i] += 1
                    rel = self.sticky_gripper_actions[i]

                if self.gripper_action_repeats[i] == self.sticky_gripper_num_repeats[i]:
                    self.sticky_action_is_ons[i]   = False
                    self.gripper_action_repeats[i] = 0
                    self.sticky_gripper_actions[i] = 0.0

                action["gripper"] = rel

            action["terminate_episode"] = np.array([0.0])
            actions_list.append(action)

        return actions_list

    def __del__(self):
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
