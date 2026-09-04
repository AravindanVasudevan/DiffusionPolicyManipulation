import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

import itertools

GRID_SPACING = 0.15 

SHAPES = ["cube", "cuboid", "cylinder"]
COLORS = {
    "red":   (0.8, 0.1, 0.1),
    "green": (0.1, 0.7, 0.2),
    "blue":  (0.1, 0.3, 0.8),
}
OBJECT_NAMES = [f"object_{shape}_{color}" for shape, color in itertools.product(SHAPES, COLORS)]

def _spawn_cfg(shape: str, color: tuple):

    material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=material,
    )
    if shape == "cube":
        return sim_utils.CuboidCfg(size=(0.045, 0.045, 0.045), **common)
    if shape == "cuboid":
        return sim_utils.CuboidCfg(size=(0.09, 0.045, 0.065), **common)
    if shape == "cylinder":
        return sim_utils.CylinderCfg(radius=0.02, height=0.05, **common)

def make_record_camera(width: int = 640, height: int = 480, fps: float = 30.0) -> CameraCfg:
    
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/record_cam",
        update_period=1.0 / fps,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(1.8, 1.2, 1.2), convention="world"),
    )


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene for the pick and place environment"""


    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    #table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
    )

    #Franka robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    #target
    target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetBin",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.3, 0.0)),
    )

    #camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=1/30.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), convention="ros"),
    )

    #lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )

    # 9 pickable objects: cube / cuboid / cylinder, each in red / yellow / blue.
    def __post_init__(self):

        combos = list(itertools.product(SHAPES, COLORS.items()))
        base_x, base_y, base_z = 0.5, -0.15, 0.05

        for idx, (shape, (color_name, rgb)) in enumerate(combos):
            name = f"object_{shape}_{color_name}"
            row, col = divmod(idx, 3)
            pos = (base_x + row * GRID_SPACING, base_y + col * GRID_SPACING, base_z)

            setattr(self, name, RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=_spawn_cfg(shape, rgb),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            ))