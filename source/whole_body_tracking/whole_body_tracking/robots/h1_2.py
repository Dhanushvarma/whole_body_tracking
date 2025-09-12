import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

from .g1 import ARMATURE_5020, ARMATURE_7520_14, ARMATURE_7520_22, ARMATURE_4010

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# Stiffness calculations
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

# Damping calculations
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

H1_2_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/unitree_description/urdf/h1_2/main.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),  # Adjusted height for H1_2
        # TODO(dhanush): Ask Niraj what the default joint pos should be.
        joint_pos={
            # Leg joints - neutral standing position
            ".*_hip_pitch_joint": -0.25,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.25,
            ".*_ankle_roll_joint": 0.0,
            # Torso
            "torso_joint": 0.0,
            # Arms - relaxed position
            "left_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.15,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.6,
            "right_shoulder_pitch_joint": 0.3,
            "right_shoulder_roll_joint": -0.15,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.6,
            # Wrists - neutral
            ".*_wrist_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 300.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 23.0,
                ".*_hip_roll_joint": 23.0,
                ".*_hip_pitch_joint": 23.0,
                ".*_knee_joint": 14.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 1.438848921 * STIFFNESS_7520_22,
                ".*_hip_roll_joint": 1.438848921 * STIFFNESS_7520_22,
                ".*_hip_pitch_joint": 1.438848921 * STIFFNESS_7520_22,
                ".*_knee_joint": 2.158273381 * STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_yaw_joint": 1.438848921 * DAMPING_7520_22,
                ".*_hip_roll_joint": 1.438848921 * DAMPING_7520_22,
                ".*_hip_pitch_joint": 1.438848921 * DAMPING_7520_22,
                ".*_knee_joint": 2.158273381 * DAMPING_7520_22,
            },
            armature={
                ".*_hip_yaw_joint": 1.438848921 * ARMATURE_7520_22,
                ".*_hip_roll_joint": 1.438848921 * ARMATURE_7520_22,
                ".*_hip_pitch_joint": 1.438848921 * ARMATURE_7520_22,
                ".*_knee_joint": 2.158273381 * ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 40.0,
            },
            velocity_limit_sim=9.0,
            stiffness={
                ".*_ankle_pitch_joint": 2.4 * STIFFNESS_5020,
                ".*_ankle_roll_joint": 1.6 * STIFFNESS_5020,
            },
            damping={
                ".*_ankle_pitch_joint": 2.4 * DAMPING_5020,
                ".*_ankle_roll_joint": 1.6 * DAMPING_5020,
            },
            armature={
                ".*_ankle_pitch_joint": 2.4 * ARMATURE_5020,
                ".*_ankle_roll_joint": 1.6 * ARMATURE_5020,
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=23.0,
            stiffness=1.438848921 * STIFFNESS_7520_22,
            damping=1.438848921 * DAMPING_7520_22,
            armature=1.438848921 * ARMATURE_7520_22,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim=40.0,
            velocity_limit_sim=9.0,
            stiffness=1.6 * STIFFNESS_5020,
            damping=1.6 * DAMPING_5020,
            armature=1.6 * ARMATURE_5020,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=18.0,
            velocity_limit_sim=20.0,
            stiffness=0.75 * STIFFNESS_5020,
            damping=0.75 * DAMPING_5020,
            armature=0.75 * ARMATURE_5020,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=19.0,
            velocity_limit_sim=31.4,
            stiffness=0.75 * STIFFNESS_5020,
            damping=0.75 * DAMPING_5020,
            armature=0.75 * ARMATURE_5020,
        ),
    },
)

# Action scale calculation for H1_2
H1_2_ACTION_SCALE = {}
for a in H1_2_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            H1_2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
