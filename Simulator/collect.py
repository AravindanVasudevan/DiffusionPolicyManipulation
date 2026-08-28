import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect scripted pick-and-place demonstrations.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel envs to collect with.")
parser.add_argument("--num_demos", type=int, default=50, help="Number of successful demos to collect.")
parser.add_argument("--dataset_dir", type=str, default="Dataset", help="Output directory for the HDF5 dataset.")
parser.add_argument("--dataset_filename", type=str, default="pick_place_demos", help="Output HDF5 file name (no extension).")
parser.add_argument("--max_steps", type=int, default=200_000, help="Safety cap on total env.step() calls.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from Simulator.env import PickPlaceEnvCfg
from Simulator.scene import OBJECT_NAMES
from Simulator import mdp

WAYPOINT_DWELL = [40, 30, 20, 30, 40, 25, 15, 30]
NUM_WAYPOINTS = len(WAYPOINT_DWELL)


def compute_waypoints(object_pos: torch.Tensor, bin_pos: torch.Tensor, device):
    """Return the 8 target (position, quat, gripper) waypoints for pick-and-place.

    object_pos / bin_pos: (num_envs, 3) positions in the robot-root frame. gripper sign
    follows BinaryJointPositionAction's convention: negative = close, non-negative = open.

    IK targets the panda_hand frame, not the fingertips, so every height here is offset by
    TCP_OFFSET (the ~0.1034m gap from panda_hand to the finger pinch point) -- otherwise the
    commanded grasp/place height puts the *hand* where the *fingertips* should be, and the
    hand jams into the table/bin well above the intended contact height.
    """
    TCP_OFFSET = 0.1034
    approach_height = 0.15 + TCP_OFFSET
    grasp_offset = 0.0 + TCP_OFFSET
    lift_height = 0.20 + TCP_OFFSET
    place_height = 0.15 + TCP_OFFSET

    down_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    OPEN, CLOSE = 1.0, -1.0

    waypoints = [
        (object_pos + torch.tensor([0, 0, approach_height], device=device), down_quat, OPEN),
        (object_pos + torch.tensor([0, 0, grasp_offset], device=device), down_quat, OPEN),
        (object_pos + torch.tensor([0, 0, grasp_offset], device=device), down_quat, CLOSE),
        (object_pos + torch.tensor([0, 0, lift_height], device=device), down_quat, CLOSE),
        (bin_pos + torch.tensor([0, 0, place_height], device=device), down_quat, CLOSE),
        (bin_pos + torch.tensor([0, 0, grasp_offset + 0.02], device=device), down_quat, CLOSE),
        (bin_pos + torch.tensor([0, 0, grasp_offset + 0.02], device=device), down_quat, OPEN),
        (bin_pos + torch.tensor([0, 0, place_height], device=device), down_quat, OPEN),
    ]
    return waypoints


def main():

    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    cfg.episode_length_s = 10.0

    cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cfg.terminations.success = DoneTerm(func=mdp.object_in_target_bin, params={"object_names": OBJECT_NAMES})

    cfg.recorders = ActionStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args_cli.dataset_dir
    )
    cfg.recorders.dataset_filename = args_cli.dataset_filename
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = ManagerBasedRLEnv(cfg=cfg)
    device = env.device

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(env.scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=env.num_envs,
        device=device,
    )
    default_arm_joint_pos = robot.data.default_joint_pos[:, robot_entity_cfg.joint_ids].clone()

    bin_pos_b = torch.tensor([0.6, 0.3, 0.0], device=device).expand(env.num_envs, 3)

    all_positions = torch.zeros(env.num_envs, NUM_WAYPOINTS, 3, device=device)
    all_grippers = torch.zeros(env.num_envs, NUM_WAYPOINTS, device=device)
    down_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(env.num_envs, 4)

    def capture_waypoints(env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        positions = torch.stack(
            [env.scene[name].data.root_pos_w for name in OBJECT_NAMES], dim=1
        )[env_ids]
        object_idx = env.command_manager.get_term("pick_place_command").object_idx[env_ids]
        object_pos_w = positions[torch.arange(len(env_ids), device=device), object_idx]
        object_pos_b = object_pos_w - robot.data.root_pos_w[env_ids]

        waypoints = compute_waypoints(object_pos_b, bin_pos_b[env_ids], device)
        for k, (pos, _quat, gripper) in enumerate(waypoints):
            all_positions[env_ids, k] = pos
            all_grippers[env_ids, k] = gripper

    SETTLE_STEPS = 15
    settle_countdown = torch.full((env.num_envs,), SETTLE_STEPS, dtype=torch.long, device=device)

    phase = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    phase_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    dwell = torch.tensor(WAYPOINT_DWELL, device=device)

    env.reset()

    step_count = 0
    last_reported = 0
    arange_envs = torch.arange(env.num_envs, device=device)
    while simulation_app.is_running() and step_count < args_cli.max_steps:
        settling = settle_countdown > 0

        target_pos_b = all_positions[arange_envs, phase]
        gripper_cmd = all_grippers[arange_envs, phase].unsqueeze(-1)
        ik_controller.set_command(torch.cat([target_pos_b, down_quat], dim=-1))

        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        arm_action = torch.where(settling.unsqueeze(-1), torch.zeros_like(joint_pos_des), joint_pos_des - default_arm_joint_pos)
        gripper_cmd = torch.where(settling.unsqueeze(-1), torch.ones_like(gripper_cmd), gripper_cmd)
        action = torch.cat([arm_action, gripper_cmd], dim=-1)

        env.step(action)
        step_count += 1

        settle_countdown = torch.clamp(settle_countdown - 1, min=0)
        just_settled = (settle_countdown == 0) & settling
        capture_waypoints(just_settled.nonzero(as_tuple=False).squeeze(-1))

        phase_steps += torch.where(settling, torch.zeros_like(phase_steps), torch.ones_like(phase_steps))
        advance = (~settling) & (phase_steps >= dwell[phase])
        phase = torch.where(advance & (phase < NUM_WAYPOINTS - 1), phase + 1, phase)
        phase_steps = torch.where(advance, torch.zeros_like(phase_steps), phase_steps)

        reset_ids = env.termination_manager.dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            phase[reset_ids] = 0
            phase_steps[reset_ids] = 0
            settle_countdown[reset_ids] = SETTLE_STEPS

        done_count = env.recorder_manager.exported_successful_episode_count
        if done_count != last_reported:
            failed_count = env.recorder_manager.exported_failed_episode_count
            print(f"[collect] {done_count}/{args_cli.num_demos} successful demos recorded "
                  f"(step {step_count}, {failed_count} discarded)")
            last_reported = done_count
        if done_count >= args_cli.num_demos:
            break

    dataset_path = os.path.join(cfg.recorders.dataset_export_dir_path, cfg.recorders.dataset_filename) + ".hdf5"
    print(f"Done. {last_reported} demos saved to {dataset_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
