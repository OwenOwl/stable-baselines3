import multiprocessing as mp
import os
from copy import deepcopy
from functools import partial
from multiprocessing.connection import Connection
from typing import Callable, List, Optional, Sequence, Type, Union

import numpy as np
import sapien.core as sapien
import torch
from gym import spaces, Wrapper
from gym.spaces import Dict as GymDict

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.real_world import lab
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def find_available_port():
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
        server_address = f"localhost:{port}"
    return server_address


def _worker(
        rank: int,
        remote: Connection,
        parent_remote: Connection,
        env_fn: Callable[..., BaseRLEnv],
):
    # NOTE(jigu): Set environment variables for ManiSkill2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    parent_remote.close()

    env = None
    try:
        env = env_fn()
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    # info["terminal_observation"] = obs
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset(seed=data)
                remote.send(obs)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "handshake":
                remote.send(None)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    except KeyboardInterrupt:
        print("Worker KeyboardInterrupt")
    except EOFError:
        print("Worker EOF")
    except Exception as err:
        print(err)
    finally:
        if env is not None:
            env.close()


class HandTeleopVecEnv(VecEnv):
    device: torch.device
    remotes: List[Connection] = []
    work_remotes: List[Connection] = []
    processes: List[mp.Process] = []

    def __init__(
            self,
            env_fns: List[Callable[[], BaseRLEnv]],
            start_method: Optional[str] = None,
            server_address: str = "auto",
            server_kwargs: dict = None,
    ):
        self.waiting = False
        self.closed = False

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        remote, work_remote = ctx.Pipe()
        args = (0, work_remote, remote, env_fns[0])
        process = ctx.Process(target=_worker, args=args, daemon=True)
        process.start()
        work_remote.close()
        remote.send(("get_attr", "observation_space"))
        observation_space: spaces.Dict = remote.recv()
        remote.send(("get_attr", "action_space"))
        action_space: spaces.Space = remote.recv()
        remote.send(("close", None))
        remote.close()
        process.join()
        # ---------------------------------------------------------------------------- #

        n_envs = len(env_fns)
        self.num_envs = n_envs

        # Allocate numpy buffers
        self.non_image_obs_space = deepcopy(observation_space)
        state_key_names = ["oracle_state", "state"]
        self.visual_obs_space = GymDict()

        camera_names = []
        for key in observation_space.keys():
            if key not in state_key_names:
                self.visual_obs_space.spaces.update({key: self.non_image_obs_space.spaces.pop(key)})
                camera_names.append(str(key).split("-")[0])
        self._last_obs_np = [None for _ in range(n_envs)]
        self._obs_np_buffer = create_np_buffer(self.non_image_obs_space, n=n_envs)

        # Start RenderServer
        if server_address == "auto":
            server_address = find_available_port()
        self.server_address = server_address
        server_kwargs = {} if server_kwargs is None else server_kwargs
        self.server = sapien.RenderServer(**server_kwargs, do_not_load_texture=True)
        self.server.start(self.server_address)
        print(f"RenderServer is running at: {server_address}")

        # Wrap env_fn
        for i, env_fn in enumerate(env_fns):
            client_kwargs = {"address": self.server_address, "process_index": i}
            env_fns[i] = partial(
                env_fn, renderer="client", renderer_kwargs=client_kwargs
            )

        # Initialize workers
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for rank in range(n_envs):
            work_remote = self.work_remotes[rank]
            remote = self.remotes[rank]
            env_fn = env_fns[rank]
            args = (rank, work_remote, remote, env_fn)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process: mp.Process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        # To make sure environments are initialized in all workers
        for remote in self.remotes:
            remote.send(("handshake", None))
        for remote in self.remotes:
            remote.recv()

        # Infer texture names
        texture_names = set()
        for visual_key in self.visual_obs_space.keys():
            if "point_cloud" in visual_key:
                texture_names.add("Position")
            else:
                raise NotImplementedError

        self.texture_names = tuple(texture_names)
        self._obs_torch_buffer: List[torch.Tensor] = self.server.auto_allocate_torch_tensors(self.texture_names)
        self.device = self._obs_torch_buffer[0].device

        # Camera pose
        # TODO: assume all environment has the same camera pose
        camera_poses = []
        for camera_name in camera_names:
            self.remotes[0].send(("env_method", ("get_camera_to_robot_pose", [camera_name], {})))
            camera_poses.append(torch.tensor(self.remotes[0].recv()))
        self.camera_rot = torch.stack([p[:3, :3].T for p in camera_poses], dim=0).to(self.device)
        self.camera_rot = torch.tile(self.camera_rot, [1, self.num_envs, 1, 1])  # [N, B, 3, 3]
        self.camera_pos = torch.stack([p[:3, 3:4].T for p in camera_poses], dim=0).to(self.device)
        self.camera_pos = torch.tile(self.camera_pos, [1, self.num_envs, 1, 1])  # [N, B, 3, 1]

        super().__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

    # ---------------------------------------------------------------------------- #
    # Observations
    # ---------------------------------------------------------------------------- #
    def _update_np_buffer(self, obs_list, indices=None):
        indices = self._get_indices(indices)
        for i, obs in zip(indices, obs_list):
            self._last_obs_np[i] = obs
        return stack_obs(
            self._last_obs_np, self.non_image_obs_space, self._obs_np_buffer
        )

    def process_pc(self, point_cloud: torch.Tensor, cam_id):
        # point_cloud: [B, H, W, 4]
        cam_rot, cam_pos = self.camera_rot[cam_id], self.camera_pos[cam_id]
        b, h, w, _ = point_cloud.shape
        point_cloud = torch.reshape(point_cloud[..., :3], [b, -1, 3])  # [B, N, 3]
        point_cloud = torch.bmm(point_cloud, cam_rot) + cam_pos
        point_cloud = batch_process_relocate_pc(point_cloud, 512)
        return point_cloud

    @torch.no_grad()
    def _get_torch_observations(self):
        self.server.wait_all()

        # TODO: only point cloud are supported for now
        tensor_dict = {}
        for i, name in enumerate(self.texture_names):
            tensor_dict[name] = self._obs_torch_buffer[i]

        image_obs = {}
        tex_name = self.texture_names[0]
        for cam_id, obs_name in enumerate(self.visual_obs_space.spaces.keys()):
            tensor = tensor_dict[tex_name][:, cam_id]  # [B, H, W, C]
            if "point_cloud" in obs_name:
                image_obs[obs_name] = self.process_pc(tensor, cam_id)

        return image_obs

    def reset_async(
            self,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
            indices=None,
    ):
        remotes = self._get_target_remotes(indices)
        if seed is None:
            seed = [None for _ in range(len(remotes))]
        if isinstance(seed, int):
            seed = [seed + i for i in range(len(remotes))]
        for remote, single_seed in zip(remotes, seed):
            remote.send(("reset", single_seed))
        self.waiting = True

    def reset_wait(
            self,
            timeout: Optional[Union[int, float]] = None,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            indices=None,
    ):
        remotes = self._get_target_remotes(indices)
        results = [remote.recv() for remote in remotes]
        self.waiting = False
        vec_obs = self._get_torch_observations()
        self._update_np_buffer(results, indices)
        vec_obs.update(deepcopy(self._obs_np_buffer))
        return vec_obs

    def reset(
            self,
            *,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
            indices=None,
    ):
        self.reset_async(seed=seed, options=options, indices=indices)
        return self.reset_wait(seed=seed, options=options, indices=indices)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_list, rews, done, infos = zip(*results)
        vec_obs = self._get_torch_observations()
        self._update_np_buffer(obs_list)
        vec_obs.update(deepcopy(self._obs_np_buffer))
        return (
            vec_obs,
            np.array(rews),
            np.array(done, dtype=np.bool_),
            infos,
        )

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self):
        raise NotImplementedError

    def get_attr(self, attr_name: str, indices=None) -> List:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(
            self,
            method_name: str,
            *method_args,
            indices=None,
            **method_kwargs,
    ) -> List:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(
            self, wrapper_class: Type[Wrapper], indices=None
    ) -> List[bool]:
        return [False] * self.num_envs

    @property
    def unwrapped(self) -> "HandTeleopVecEnv":
        return self

    def _get_indices(self, indices) -> List[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]
        return indices

    def _get_target_remotes(self, indices) -> List[Connection]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, self.env_method("__repr__", indices=0)[0]
        )

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        remotes = self.remotes
        if seed is None:
            seed = [None for _ in range(len(remotes))]
        if isinstance(seed, int):
            seed = [seed + i for i in range(len(remotes))]
        for remote, single_seed in zip(remotes, seed):
            remote.send(("reset", single_seed))
        _ = [remote.recv() for remote in self.remotes]
        self.waiting = True
        return seed


def create_np_buffer(space: spaces.Space, n: int):
    if isinstance(space, spaces.Dict):
        return {
            key: create_np_buffer(subspace, n) for key, subspace in space.spaces.items()
        }
    elif isinstance(space, spaces.Box):
        return np.zeros((n,) + space.shape, dtype=space.dtype)
    else:
        raise NotImplementedError(
            "Unsupported observation space: {}".format(type(space))
        )


def stack_obs(obs: Sequence, space: spaces.Space, buffer: Optional[np.ndarray] = None):
    if isinstance(space, spaces.Dict):
        ret = {}
        for key in space:
            _obs = [o[key] for o in obs]
            _buffer = None if buffer is None else buffer[key]
            ret[key] = stack_obs(_obs, space[key], buffer=_buffer)
        return ret
    elif isinstance(space, spaces.Box):
        if not isinstance(obs[0], np.ndarray):  # check for 0-dimensional parameter
            obs = [
                np.array(o)[None] for o in obs
            ]  # convert float to array and add dimension
        return np.stack(obs, out=buffer)
    else:
        raise NotImplementedError(type(space))


def batch_index_select(input_tensor, index, dim):
    """Batch index_select
    Code Source: https://github.com/Jiayuan-Gu/torkit3d/blob/master/torkit3d/nn/functional.py
    Args:
        input_tensor (torch.Tensor): [B, ...]
        index (torch.Tensor): [B, N] or [B]
        dim (int): the dimension to index
    References:
        https://discuss.pytorch.org/t/batched-index-select/9115/7
        https://github.com/vacancy/AdvancedIndexing-PyTorch
    """

    if index.dim() == 1:
        index = index.unsqueeze(1)
        squeeze_dim = True
    else:
        assert (
                index.dim() == 2
        ), "index is expected to be 2-dim (or 1-dim), but {} received.".format(
            index.dim()
        )
        squeeze_dim = False
    assert input_tensor.size(0) == index.size(0), "Mismatched batch size: {} vs {}".format(
        input_tensor.size(0), index.size(0)
    )
    views = [1 for _ in range(input_tensor.dim())]
    views[0] = index.size(0)
    views[dim] = index.size(1)
    expand_shape = list(input_tensor.shape)
    expand_shape[dim] = -1
    index = index.view(views).expand(expand_shape)
    out = torch.gather(input_tensor, dim, index)
    if squeeze_dim:
        out = out.squeeze(1)
    return out


def batch_process_relocate_pc(pc: torch.Tensor, num_points: int, noise_level=1):
    device = pc.device
    batch = pc.shape[0]
    bound = lab.RELOCATE_BOUND
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = within_bound_x & within_bound_y & within_bound_z
    within_indices = [torch.nonzero(within_bound[i]) for i in range(batch)]
    index_list = []
    noise = []
    for i in range(batch):
        indices = within_indices[i][:, 0]
        num_index = len(indices)
        if num_index == 0:
            pc[i, 0] = torch.zeros([3], dtype=pc.dtype, device=device)
            indices = torch.zeros(num_points, dtype=indices.dtype, device=device)
            multiplicative_noise = torch.ones([num_points, 1], device=device)
        elif num_index < num_points:
            indices = torch.cat(
                [indices, torch.zeros(num_points - num_index, dtype=indices.dtype, device=device)])
            multiplicative_noise = 1 + torch.randn(num_index, device=device)[:,
                                       None] * 0.01 * noise_level  # (num_index, 1)
            multiplicative_noise = torch.concat([multiplicative_noise,
                                                 torch.ones([num_points - num_index, 1], device=device) *
                                                 multiplicative_noise[0]], axis=0)
        else:
            indices = indices[torch.randperm(num_index, device=device)[:num_points]]
            multiplicative_noise = 1 + torch.randn(num_points, device=device)[:, None] * 0.01 * noise_level  # (n, 1)
        index_list.append(indices)
        noise.append(multiplicative_noise)

    noise = torch.stack(noise).to(device)
    batch_indices = torch.stack(index_list)
    batch_cloud = batch_index_select(pc, batch_indices, dim=1) * noise
    return batch_cloud
