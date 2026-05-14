import json
import os
import re
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from .trace_processor import TraceProcessor

# ---------------------------------------------------------------------------
# Grounding DINO – optional dependency.  Foveation is silently skipped when
# the package is not installed or the checkpoint is not provided.
# ---------------------------------------------------------------------------
try:
    from groundingdino.util.inference import load_model as _gdino_load, predict as _gdino_predict
    import torchvision.transforms as T

    _GDINO_AVAILABLE = True
except ImportError:
    _GDINO_AVAILABLE = False


def _load_gdino(config_path: str, checkpoint_path: str, device):
    """Return a Grounding DINO model, or None if unavailable."""
    if not _GDINO_AVAILABLE:
        return None
    if not (os.path.isfile(config_path) and os.path.isfile(checkpoint_path)):
        return None
    return _gdino_load(config_path, checkpoint_path, device=str(device))


def _extract_noun_phrase(task_description: str) -> str:
    """
    Heuristically pull the target-object noun phrase from a task description.
    E.g. 'pick up the blue cup' → 'blue cup'.
    Falls back to the full description if no pattern matches.
    """
    patterns = [
        r"(?:pick up|grasp|grab|move|place|put|push|open|close|lift|carry)\s+(?:the\s+)?(.+?)(?:\s+(?:into|onto|on|to|from|in|and)\b|$)",
        r"(?:the\s+)(.+?)(?:\s+(?:into|onto|on|to|from|in|and)\b|$)",
    ]
    desc = task_description.lower().strip().rstrip("?.")
    for pat in patterns:
        m = re.search(pat, desc)
        if m:
            return m.group(1).strip()
    return desc


_GDINO_TRANSFORM = None


def _gdino_transform():
    global _GDINO_TRANSFORM
    if _GDINO_TRANSFORM is None:
        _GDINO_TRANSFORM = T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _GDINO_TRANSFORM


def _get_fovea_bbox(
    gdino_model,
    image_pil: Image.Image,
    caption: str,
    image_size: int = 224,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    device="cuda",
) -> Optional[torch.Tensor]:
    """
    Run Grounding DINO and return the highest-confidence bbox as a [1, 4] pixel-space xyxy tensor,
    or None if nothing is detected.
    """
    if gdino_model is None:
        return None
    image_tensor = _gdino_transform()(image_pil).to(device)
    boxes, logits, _ = _gdino_predict(
        gdino_model, image_tensor, caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if len(boxes) == 0:
        return None
    # boxes: [N, 4] cxcywh normalised → convert to pixel-space xyxy
    best = boxes[logits.argmax()]  # [4] cx, cy, w, h  in [0,1]
    cx, cy, w, h = best.unbind(-1)
    x1 = (cx - w / 2) * image_size
    y1 = (cy - h / 2) * image_size
    x2 = (cx + w / 2) * image_size
    y2 = (cy + h / 2) * image_size
    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.
    Using tf resize corresponding to RLDS resize.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = Image.fromarray(img.numpy())
    return img

class TraceVLAInference:
    """
    For future developers:
    - Refer to BaseVectorPolicy for the API contract
    """

    def __init__(
        self,
        model_path,
        cotracker_model_path,
        dataset_stats_path,
        action_scale: float = 1.0,
        n_action_bins: int = 256,
        sample: bool = False,
        temperature: float = 0.0,
        image_aug: bool = False,
        model_dtype: str = None,  # in ["bfloat16", "float16", "float32", None]; None will read from the model config
        device: int = 0,
        # --- Latent Foveation ---
        gdino_config_path: Optional[str] = None,
        gdino_checkpoint_path: Optional[str] = None,
        fovea_scale: float = 1.2,
        secondary_scale: float = 0.7,
        bg_scale: float = 0.4,
        gdino_box_threshold: float = 0.30,
        gdino_text_threshold: float = 0.25,
    ) -> None:
        self.cotracker_model_path = cotracker_model_path
        self.device = device

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            center_crop=False
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device=device)
        print('Instantiated TraceVLA model')

        # --- Latent Foveation setup ---
        self.gdino_model = _load_gdino(gdino_config_path or "", gdino_checkpoint_path or "", device)
        self.gdino_box_threshold = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold
        if self.gdino_model is not None:
            self.vla.configure_foveation(
                image_size=224,
                patch_size=14,
                fovea_scale=fovea_scale,
                secondary_scale=secondary_scale,
                bg_scale=bg_scale,
            )
            print("Latent Foveation enabled (Grounding DINO loaded)")
        else:
            print("Latent Foveation disabled (Grounding DINO not available)")
        
        with open(dataset_stats_path, "r") as f:
            self.norm_stats = json.load(f)
        
        from cotracker.predictor import CoTrackerPredictor
        self.cotracker_model = CoTrackerPredictor(
            checkpoint=os.path.join(
                '/mnt/amlfs-01/home/ruijiez/co-tracker/checkpoints/scaled_offline.pth'
                )
        ).to(device)
            
        self.prompt_template = "In: What action should the robot take to {task_description}?\nOut:"
        self.prompt_overlaid_template = "In: You are given two images: one with the original robot observation, and another one marked with historical traces of the robot end effector and moving objects, separated by a special separator token. What action should the robot take to {task_description}?\nOut:"
        self.action_scale = action_scale
        self.sample = sample
        self.temperature = temperature
        self.image_aug = image_aug

        self.sticky_action_is_ons = [False]
        self.gripper_action_repeats = [0]
        self.sticky_gripper_actions = [0.0]
        self.previous_gripper_actions = [None]

    def start(self, num_envs: int):
        self.num_envs = num_envs

        # policy setups
        self.policy_setups = [None] * self.num_envs
        self.sticky_gripper_num_repeats = [0] * self.num_envs
        self.unnorm_keys = [None] * self.num_envs

        # sticky actions
        self.sticky_action_is_ons = [False] * self.num_envs
        self.gripper_action_repeats = [0] * self.num_envs
        self.sticky_gripper_actions = [0.0] * self.num_envs
        self.previous_gripper_actions = [None] * self.num_envs

        # task descriptions
        self.task_descriptions = [None] * self.num_envs

    def start_episode(
        self,
        obs_dicts: List[Dict],
        info_dicts: List[Dict],
        output_dirs: Optional[List[str]] = None,
        ids: Optional[Union[List[int], np.ndarray]] = None,
    ):
        ids = self.unwrap_ids(ids)
        policy_setups = get_batch_from_info(info_dicts, "policy_setup")
        task_descriptions = get_batch_from_info(info_dicts, "task_description")

        assert (
            len(obs_dicts) == len(info_dicts) == len(output_dirs) == len(ids)
        ), f"Mismatch in lengths: {len(obs_dicts)=}, {len(info_dicts)=}, {len(output_dirs)=}, {len(ids)=}"

        for i, (policy_setup, task_description) in enumerate(zip(policy_setups, task_descriptions)):
            idx = ids[i]
            self.policy_setups[idx] = policy_setup
            if policy_setup == "widowx_bridge":
                self.sticky_gripper_num_repeats[idx] = 1
                self.unnorm_keys[idx] = "bridge_orig"
            elif policy_setup == "google_robot":
                self.sticky_gripper_num_repeats[idx] = 15
                self.unnorm_keys[idx] = "fractal20220817_data"

            self.sticky_action_is_ons[idx] = False
            self.gripper_action_repeats[idx] = 0
            self.sticky_gripper_actions[idx] = 0.0
            self.previous_gripper_actions[idx] = None
            self.task_descriptions[idx] = task_description
            
            self.trace_processors = []
            for i in range(len(policy_setups)):
                self.trace_processor = TraceProcessor(cotracker_model=self.cotracker_model_path, 
                                                      window_size=15 if policy_setups[i] == 'google_robot' else 10,
                                                      device=self.vla.device,
                                                     )
                self.trace_processor.reset()
                self.trace_processors.append(self.trace_processor)

    def reset_states_at(self, idx, task_description):
        assert idx < self.num_envs, f"{idx=}, {self.num_envs=}"
        self.sticky_action_is_ons[idx] = False
        self.gripper_action_repeats[idx] = 0
        self.sticky_gripper_actions[idx] = 0.0
        self.previous_gripper_actions[idx] = None
        self.task_descriptions[idx] = task_description
        self.trace_processors[idx].reset()

    @torch.no_grad()
    def get_action(
        self,
        obs_dicts: List[Dict],
        info_dicts: List[Dict],
        ids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        ids = self.unwrap_ids(ids)
        assert len(ids) == self.num_envs, f"Expected {self.num_envs} ids, but got {len(ids)}"

        images = get_batch_from_info(info_dicts, "image")
        task_descriptions = get_batch_from_info(info_dicts, "task_description")

        # Compare and reset task descriptions if needed
        for i, task_description in enumerate(task_descriptions):
            if task_description != self.task_descriptions[i]:
                self.reset_states_at(i, task_description)

        assert len(images) == self.num_envs, f"{len(images)=}, {self.num_envs=}"
        
        raw_actions = []
        for i in range(self.num_envs):
            image = images[i]
            image = Image.fromarray(image)
            image = resize_image(image, (256,256))
            image_overlaid, has_trace = self.trace_processors[i].process_image(image)
            
            task_description = self.task_descriptions[i]
            
            if not has_trace:
                prompt = self.prompt_template.format(task_description=task_description)
                inputs = self.processor(prompt, [image, image]).to(device=self.vla.device, dtype=torch.bfloat16)
            else:
                prompt = self.prompt_overlaid_template.format(task_description=task_description)
                inputs = self.processor(prompt, [image, image_overlaid]).to(device=self.vla.device, dtype=torch.bfloat16)

            # Latent Foveation: ground target object and compute fovea bbox
            fovea_bbox = None
            if self.gdino_model is not None:
                caption = _extract_noun_phrase(task_description)
                fovea_bbox = _get_fovea_bbox(
                    self.gdino_model, image, caption,
                    image_size=224,
                    box_threshold=self.gdino_box_threshold,
                    text_threshold=self.gdino_text_threshold,
                    device=self.vla.device,
                )
                if fovea_bbox is not None:
                    fovea_bbox = fovea_bbox.to(device=self.vla.device, dtype=torch.bfloat16)

            with torch.inference_mode():
                raw_action = self.vla.predict_action(
                    **inputs,
                    unnorm_key=self.unnorm_keys[i],
                    do_sample=self.sample,
                    temperature=0.7,
                    fovea_bbox=fovea_bbox,
                )
                raw_actions.append(raw_action)
        
        
        actions_list = []
        for i in range(self.num_envs):
            actions = raw_actions[i]
            raw_action = {
                "world_vector": np.array(actions[:3]),
                "rotation_delta": np.array(actions[3:6]),
                "open_gripper": np.array(actions[6:7]),
            }
            action = {}
            action["world_vector"] = raw_action["world_vector"] * self.action_scale
            action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            roll, pitch, yaw = action_rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            action["rot_axangle"] = action_rotation_axangle * self.action_scale

            if self.policy_setups[i] == "widowx_bridge":
                action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            elif self.policy_setups[i] == "google_robot":
                current_gripper_action = raw_action["open_gripper"]
                if self.previous_gripper_actions[i] is None:
                    relative_gripper_action = np.array([0])
                else:
                    relative_gripper_action = (
                        self.previous_gripper_actions[i] - current_gripper_action
                    )
                self.previous_gripper_actions[i] = current_gripper_action

                if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_ons[i]):
                    self.sticky_action_is_ons[i] = True
                    self.sticky_gripper_actions[i] = relative_gripper_action

                if self.sticky_action_is_ons[i]:
                    self.gripper_action_repeats[i] += 1
                    relative_gripper_action = self.sticky_gripper_actions[i]

                if self.gripper_action_repeats[i] == self.sticky_gripper_num_repeats[i]:
                    self.sticky_action_is_ons[i] = False
                    self.gripper_action_repeats[i] = 0
                    self.sticky_gripper_actions[i] = 0.0

                action["gripper"] = relative_gripper_action

            action["terminate_episode"] = np.array([0.0])

            actions_list.append(action)

        return actions_list