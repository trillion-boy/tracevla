#!/usr/bin/env python3
"""
tracevla_latent_saccade_eval.py

Evaluate TraceVLA + LatentSaccade on SimplerEnv WidowX tasks.

Usage:
    python scripts/tracevla_latent_saccade_eval.py \
        --model-path      /path/to/tracevla \
        --dataset-stats   /path/to/dataset_statistics.json \
        --cotracker-ckpt  /path/to/scaled_offline.pth \
        --task            widowx_stack_cube \
        --n-episodes      24 \
        --output-dir      /tmp/latent_saccade_results \
        --bg-weight       0.2 \
        --place-src-weight 0.5 \
        --fovea-weight    1.0 \
        --save-video

    # Baseline (no weight masking):
        --disable-latent-mask
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ── SimplerEnv paths (adjust if needed) ───────────────────────────────────────
SIMPLER_ROOT = os.environ.get("SIMPLER_ROOT", "/content/SimplerEnv")
for p in [SIMPLER_ROOT, os.path.join(SIMPLER_ROOT, "ManiSkill2_real2sim")]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from prismatic.eval.latent_saccade_tracevla import LatentSaccadeTraceVLAInference


# ── Task configs (same tasks as UniVLA standalone eval) ───────────────────────

TASK_CONFIGS = {
    "widowx_stack_cube": {
        "env_name":    "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "robot":       "widowx",
        "scene_name":  "bridge_table_1_v1",
        "rgb_overlay_path":    "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range":   [0, 24],
        "obs_camera_name":     "3rd_view_camera",
        "policy_setup":        "widowx_bridge",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 60,
    },
    "widowx_carrot_on_plate": {
        "env_name":    "PutCarrotOnPlateInScene-v0",
        "robot":       "widowx",
        "scene_name":  "bridge_table_1_v1",
        "rgb_overlay_path":    "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range":   [0, 24],
        "obs_camera_name":     "3rd_view_camera",
        "policy_setup":        "widowx_bridge",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 60,
    },
    "widowx_put_eggplant_in_basket": {
        "env_name":    "PutEggplantInBasketScene-v0",
        "robot":       "widowx_sink_camera_setup",
        "scene_name":  "bridge_table_1_v2",
        "rgb_overlay_path":    "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range":   [0, 24],
        "obs_camera_name":     "3rd_view_camera",
        "policy_setup":        "widowx_bridge",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 120,
    },
}


# ── SimplerEnv helpers ─────────────────────────────────────────────────────────

def build_env(cfg, ep_id):
    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    robot = cfg["robot"]
    kw = dict(
        obs_mode="rgbd",
        robot=robot,
        sim_freq=cfg["sim_freq"],
        control_mode=get_robot_control_mode(robot, "openvla"),
        control_freq=cfg["control_freq"],
        max_episode_steps=cfg["max_episode_steps"],
        scene_name=cfg["scene_name"],
        camera_cfgs={"add_segmentation": True},
    )
    for base in [SIMPLER_ROOT, os.path.join(SIMPLER_ROOT, "ManiSkill2_real2sim")]:
        cand = os.path.join(base, cfg["rgb_overlay_path"])
        if os.path.exists(cand):
            kw["rgb_overlay_path"]    = cand
            kw["rgb_overlay_cameras"] = cfg["rgb_overlay_cameras"]
            break
    env = build_maniskill2_env(cfg["env_name"], **kw)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": ep_id}})
    return env, obs


def get_image(env, obs, cam_name):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    return get_image_from_maniskill2_obs_dict(env, obs, camera_name=cam_name)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TraceVLA + LatentSaccade SimplerEnv Eval")
    # model
    p.add_argument("--model-path",      required=True)
    p.add_argument("--dataset-stats",   required=True)
    p.add_argument("--cotracker-ckpt",  required=True)
    # task
    p.add_argument("--task", default="widowx_stack_cube",
                   choices=list(TASK_CONFIGS))
    p.add_argument("--n-episodes", type=int, default=24)
    p.add_argument("--output-dir",  default="/tmp/latent_saccade_results")
    # weight map
    p.add_argument("--bg-weight",         type=float, default=0.5)
    p.add_argument("--place-src-weight",  type=float, default=0.8)
    p.add_argument("--fovea-weight",      type=float, default=1.2)
    # saccade
    p.add_argument("--min-grasp-steps",   type=int,   default=15)
    p.add_argument("--min-place-steps",   type=int,   default=8)
    p.add_argument("--consec-close",      type=int,   default=3)
    p.add_argument("--consec-open",       type=int,   default=3)
    # dino
    p.add_argument("--dino-cache-steps",  type=int,   default=5)
    p.add_argument("--box-threshold",     type=float, default=0.15)
    p.add_argument("--text-threshold",    type=float, default=0.15)
    p.add_argument("--bbox-margin",       type=int,   default=2)
    # misc
    p.add_argument("--device",            type=int,   default=0)
    p.add_argument("--save-video",        action="store_true")
    p.add_argument("--disable-latent-mask", action="store_true",
                   help="Ablation: run baseline without weight masking")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    task_cfg = TASK_CONFIGS[args.task]
    cam_name = task_cfg["obs_camera_name"]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[init] Loading LatentSaccadeTraceVLAInference  task={args.task}")
    print(f"       bg={args.bg_weight}  src={args.place_src_weight}  "
          f"fovea={args.fovea_weight}  mask={'OFF (baseline)' if args.disable_latent_mask else 'ON'}")

    model = LatentSaccadeTraceVLAInference(
        model_path=args.model_path,
        cotracker_model_path=args.cotracker_ckpt,
        dataset_stats_path=args.dataset_stats,
        device=args.device,
        bg_weight=args.bg_weight,
        place_src_weight=args.place_src_weight,
        fovea_weight=args.fovea_weight,
        dino_cache_steps=args.dino_cache_steps,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        bbox_margin=args.bbox_margin,
        min_grasp_steps=args.min_grasp_steps,
        min_place_steps=args.min_place_steps,
        consecutive_close_required=args.consec_close,
        consecutive_open_required=args.consec_open,
        enable_latent_mask=not args.disable_latent_mask,
    )
    model.start(num_envs=1)

    base_ids = list(range(*task_cfg["obj_episode_range"]))
    ep_ids   = [base_ids[i % len(base_ids)] for i in range(args.n_episodes)]
    results  = []

    for ep_count, ep_id in enumerate(ep_ids):
        print(f"\n── ep {ep_count:02d} (env_id={ep_id}) ──────────────────────────")
        env, obs = build_env(task_cfg, ep_id)
        instruction = env.get_language_instruction()
        image       = get_image(env, obs, cam_name)
        print(f"   instruction: {instruction}")

        # start_episode expects list-of-dicts format
        info_dict = {
            "policy_setup":    task_cfg["policy_setup"],
            "task_description": instruction,
        }
        model.start_episode(
            obs_dicts=[{}],
            info_dicts=[info_dict],
            output_dirs=[args.output_dir],
            ids=[0],
        )

        frames = [image.copy()] if args.save_video else []
        done = truncated = False
        step = 0
        t0 = time.time()

        while not (done or truncated) and step < task_cfg["max_episode_steps"]:
            info_dict["image"] = image
            action_list = model.get_action(obs_dicts=[{}], info_dicts=[info_dict], ids=[0])
            env_action = action_list[0]

            obs, _, done, truncated, _ = env.step(
                np.concatenate([
                    env_action["world_vector"],
                    env_action["rot_axangle"],
                    env_action["gripper"],
                ])
            )
            image = get_image(env, obs, cam_name)

            # track instruction changes (some tasks change mid-episode)
            new_instr = env.get_language_instruction()
            if new_instr != instruction:
                instruction = new_instr
                info_dict["task_description"] = instruction

            if args.save_video and step % 4 == 0:
                frames.append(image.copy())
            step += 1

        elapsed = time.time() - t0
        status  = "SUCCESS" if done else "FAIL"
        print(f"   → {status}  ({step} steps, {elapsed:.1f}s)")
        env.close()

        if args.save_video and frames:
            from PIL import Image as _PIL
            vpath = os.path.join(args.output_dir, f"ep{ep_count:02d}_{status.lower()}.gif")
            pils  = [_PIL.fromarray(f) for f in frames]
            pils[0].save(vpath, save_all=True, append_images=pils[1:], loop=0, duration=100)
            print(f"   GIF saved: {vpath}")

        results.append({
            "ep": ep_count, "ep_id": ep_id,
            "success": bool(done), "steps": step, "elapsed": elapsed,
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    n_ok = sum(r["success"] for r in results)
    sr   = n_ok / len(results)
    print(f"\n{'='*52}")
    print(f"  task:          {args.task}")
    print(f"  latent_mask:   {'OFF (baseline)' if args.disable_latent_mask else 'ON'}")
    print(f"  bg={args.bg_weight}  src={args.place_src_weight}  fovea={args.fovea_weight}")
    print(f"  Success rate:  {n_ok}/{len(results)} = {sr:.1%}")
    print(f"  Avg steps:     {np.mean([r['steps'] for r in results]):.0f}")
    print(f"{'='*52}")
    for r in results:
        mark = "✓" if r["success"] else "✗"
        print(f"  {mark} ep{r['ep']:02d} (id={r['ep_id']}): {r['steps']} steps")

    summary = {
        "model": "LatentSaccadeTraceVLA",
        "task":  args.task,
        "latent_mask_enabled": not args.disable_latent_mask,
        "success_rate": sr,
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "config": {
            "bg_weight":          args.bg_weight,
            "place_src_weight":   args.place_src_weight,
            "fovea_weight":       args.fovea_weight,
            "min_grasp_steps":    args.min_grasp_steps,
            "min_place_steps":    args.min_place_steps,
            "consec_close":       args.consec_close,
            "dino_cache_steps":   args.dino_cache_steps,
        },
        "episodes": results,
    }
    save_path = os.path.join(args.output_dir, f"results_{args.task}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()
