"""Bounded simple_tag with selectable bounce or touch-penalty walls."""
import json
from pathlib import Path

import numpy as np
from gymnasium.utils import EzPickle
from mpe2._mpe_utils.core import World
from mpe2.simple_tag.simple_tag import Scenario as TagScenario
from mpe2.simple_tag_v3 import raw_env as TagEnv
from mpe2._mpe_utils.simple_env import make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

BOUNDARY_RANGE = 1.5
BOUNDARY_PENALTY = 100.0
BOUNDARY_MODE = 'bounce'
NUM_OBSTACLES = 2


def add_boundary_arguments(parser, *, defaults=True):
    """Share spelling and defaults between training and evaluation."""
    parser.add_argument('--boundary-mode', '--boundary_mode', choices=('bounce', 'punish'),
                        default=BOUNDARY_MODE if defaults else None,
                        help='simple_tag walls: bounce (default) or punish; evaluation inherits training settings')
    parser.add_argument('--boundary-range', '--boundary_range', type=float,
                        default=BOUNDARY_RANGE if defaults else None,
                        help='simple_tag square half-width (default: 1.5)')
    parser.add_argument('--boundary-penalty', '--boundary_penalty', type=float,
                        default=BOUNDARY_PENALTY if defaults else None,
                        help=f'Reward deducted per touching agent per step in punish mode (default: {BOUNDARY_PENALTY:g})')
    parser.add_argument('--num-obstacles', '--num_obstacles', type=int,
                        default=NUM_OBSTACLES if defaults else None,
                        help='simple_tag obstacle count; 0 removes obstacles (default: 2; evaluation inherits training)')


def validate_obstacles(num_obstacles):
    if not isinstance(num_obstacles, (int, np.integer)) or isinstance(num_obstacles, bool) or num_obstacles < 0:
        raise ValueError('num_obstacles must be a nonnegative integer')


def validate_boundary(boundary_mode, boundary_range, boundary_penalty):
    if boundary_mode not in ('bounce', 'punish'):
        raise ValueError('boundary_mode must be bounce or punish')
    if not np.isfinite(boundary_range) or boundary_range <= 0:
        raise ValueError('boundary_range must be finite and positive')
    if not np.isfinite(boundary_penalty) or boundary_penalty < 0:
        raise ValueError('boundary_penalty must be finite and nonnegative')


def resolve_boundary_arguments(args, model_dir):
    """Explicit evaluation flags override saved training settings, then defaults."""
    if args.env_name != 'simple_tag_v3':
        return
    settings = {}
    path = Path(model_dir) / 'run_config.json'
    if path.is_file():
        settings = json.loads(path.read_text(encoding='utf-8'))
    for name, default in (('boundary_mode', BOUNDARY_MODE), ('boundary_range', BOUNDARY_RANGE),
                          ('boundary_penalty', BOUNDARY_PENALTY), ('num_obstacles', NUM_OBSTACLES)):
        if getattr(args, name, None) is None:
            setattr(args, name, settings.get(name, default))
    validate_boundary(args.boundary_mode, args.boundary_range, args.boundary_penalty)
    validate_obstacles(args.num_obstacles)


class BoundaryWorld(World):
    def integrate_state(self, p_force):
        # Apply walls before SimpleEnv computes rewards, observations or termination.
        super().integrate_state(p_force)
        for entity in self.entities:
            if not entity.movable:
                continue
            limit = self.boundary_range - entity.size
            for axis in range(self.dim_p):
                position = entity.state.p_pos[axis]
                if abs(position) >= limit:
                    normal = 1.0 if position >= 0 else -1.0
                    entity.state.p_pos[axis] = normal * limit
                    # Do not turn an already inward velocity back toward the wall.
                    if entity.state.p_vel[axis] * normal > 0:
                        if self.boundary_mode == 'bounce':
                            entity.state.p_vel[axis] *= -1
                        else:
                            entity.state.p_vel[axis] = 0.0


class BoundaryScenario(TagScenario):
    def reset_world(self, world, np_random):
        super().reset_world(world, np_random)
        # Retain the upstream reset when it fits; small custom arenas must also
        # contain full circles (including stationary obstacles) at step zero.
        for entity in world.entities:
            limit = self.boundary_range - entity.size
            if np.any(np.abs(entity.state.p_pos) > limit):
                entity.state.p_pos = np_random.uniform(-limit, limit, world.dim_p)

    def boundary_penalty(self, agent):
        # Penalize each touching agent once per step, even at a corner.
        touching = np.any(np.abs(agent.state.p_pos) >= self.boundary_range - agent.size)
        return self.penalty_per_step if self.boundary_mode == 'punish' and touching else 0.0

    def agent_reward(self, agent, world):
        # Replace the upstream 0.9/1.0 edge penalty with the configured walls.
        collisions = sum(self.is_collision(agent, adv) for adv in self.adversaries(world))
        capture_reward = -10.0 * collisions if agent.collide else 0.0
        return capture_reward - self.boundary_penalty(agent)

    def adversary_reward(self, agent, world):
        # Preserve the shared team capture reward; penalize only this agent.
        return super().adversary_reward(agent, world) - self.boundary_penalty(agent)


class raw_env(TagEnv):
    def __init__(self, *args, boundary_range=BOUNDARY_RANGE,
                 boundary_penalty=BOUNDARY_PENALTY, boundary_mode=BOUNDARY_MODE, **kwargs):
        boundary_range = float(boundary_range)
        boundary_penalty = float(boundary_penalty)
        validate_boundary(boundary_mode, boundary_range, boundary_penalty)
        super().__init__(*args, **kwargs)
        if any(entity.size >= boundary_range for entity in self.world.entities):
            raise ValueError('boundary_range must be larger than every entity radius')
        # Retain upstream scenario configuration (curriculum, observability, etc.).
        scenario = BoundaryScenario.__new__(BoundaryScenario)
        scenario.__dict__.update(self.scenario.__dict__)
        scenario.boundary_range = boundary_range
        scenario.penalty_per_step = boundary_penalty
        scenario.boundary_mode = boundary_mode
        self.scenario = scenario
        # The upstream world adds entity type annotations, but no extra physics.
        # Preserve its state and use the local bounded integrator only here.
        self.world.__class__ = BoundaryWorld
        self.world.boundary_range = boundary_range
        self.world.boundary_mode = boundary_mode
        self.boundary_range = boundary_range
        self.boundary_penalty = boundary_penalty
        self.boundary_mode = boundary_mode
        self.reset()
        EzPickle.__init__(self, *args, boundary_range=boundary_range,
                          boundary_penalty=boundary_penalty, boundary_mode=boundary_mode, **kwargs)


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
