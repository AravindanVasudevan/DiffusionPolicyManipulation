from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from .scene import PickPlaceSceneCfg, OBJECT_NAMES
from . import mdp 


@configclass
class ObservationsCfg:
    """The observations"""

    @configclass
    class PolicyCfg(ObsGroup):
        """What the policy sees"""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pos = ObsTerm(
            func=mdp.ee_position,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )
        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb", "normalize": False},
        )
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pick_place_command"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    """The actions"""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=1.0
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )

@configclass
class CommandsCfg:
    """Object command"""

    pick_place_command: mdp.PickPlaceCommandCfg = mdp.PickPlaceCommandCfg(
        object_names=OBJECT_NAMES,
    )

@configclass
class EventsCfg:
    """Domain randomization and resets"""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_objects_pos = EventTerm(
        func=mdp.randomize_object_pos_on_table,
        mode="reset",
        params={
            "object_names": OBJECT_NAMES,
            "x_range": (0.24, 0.80),
            "y_range": (-0.40, 0.40),
            "min_separation": 0.09,
        },
    )
    
@configclass
class RewardsCfg:
    """Dummy reward"""

    dummy = RewTerm(func=mdp.is_alive, weight=0.0)


@configclass
class TerminationsCfg:
    """Events which ends the episode"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_placed = DoneTerm(
        func=mdp.object_in_target_bin,
        params={"object_names": OBJECT_NAMES},
    )
    object_dropping = DoneTerm(
        func=mdp.object_dropped,
        params={"object_names": OBJECT_NAMES, "minimum_height": -0.05}
    )

@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1 / 60
        self.episode_length_s = 5.0