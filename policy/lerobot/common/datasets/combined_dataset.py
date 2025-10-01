import pytorch_lightning as pl
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import torch
from torch.utils.data.distributed import DistributedSampler
import logging
import packaging.version
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    CODEBASE_VERSION,
    HF_LEROBOT_HOME,
    aggregate_stats,
    create_empty_dataset_info,
    DEFAULT_FEATURES,
    get_features_from_robot,
    Robot,
)
# from lerobot.common.datasets.factory import IMAGENET_STATS
# from lerobot.common.datasets.factory import resolve_delta_timestamps


class LerobotCombinedDatasetMetadata:
    def __init__(
        self,
        repo_ids: list[str],
        visual_keys_used: list[str] | None = None,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_ids = repo_ids
        self.revision = revision if revision else CODEBASE_VERSION
        self.visual_keys_used = visual_keys_used
        # self.root = Path(root) if root is not None else HF_LEROBOT_HOME / "combined_dataset"
        
        # Initialize individual dataset metadata
        self.dataset_metadatas = []
        for repo_id in self.repo_ids:
            try:
                if force_cache_sync:
                    raise FileNotFoundError
                dataset_meta = LeRobotDatasetMetadata(
                    repo_id=repo_id,
                    root=None,  # Use default HF_LEROBOT_HOME / repo_id
                    revision=self.revision,
                    force_cache_sync=force_cache_sync,
                )
                self.dataset_metadatas.append(dataset_meta)
            except Exception as e:
                logging.warning(f"Failed to load metadata for dataset {repo_id}: {e}")
                continue
        
        if not self.dataset_metadatas:
            raise ValueError("No valid datasets could be loaded")
        
        # Combine metadata from all datasets
        self.combine_metadata()

    def combine_metadata(self):
        """Combine metadata from all individual datasets"""
        # Combine info
        # BUG: combined info features cannot only use dataset 0's features since different datasets may have different features
        self.info = self._combine_info()
        
        # Combine tasks
        self.tasks, self.task_to_task_index = self._combine_tasks()
        
        # Combine episodes
        self.episodes = self._combine_episodes()
        
        # Combine stats
        self.stats = self._combine_stats()
        
        # Combine episodes_stats
        # self.episodes_stats = self._combine_episodes_stats()

    def _combine_info(self) -> dict:
        """Combine info from all datasets"""
        combined_info = {}
        total_episodes = 0
        total_frames = 0
        total_tasks = 0
        total_videos = 0
        # Use features from first dataset as base
        if self.dataset_metadatas:
            combined_info = self.dataset_metadatas[0].info.copy()
            combined_info["repo_ids"] = self.repo_ids
            combined_info["num_datasets"] = len(self.dataset_metadatas)

        #### Unify the visual keys to be used across datasets ####
        visual_keys_used = self.visual_keys_used
        # visual_keys_used = ["left_image", "ego_image", "top_image"] # TODO:set in the outer shell script
        from lerobot.common.constants import DATA_KEYS_MAPPING_HAND, DATA_KEYS_MAPPING_ROBOT
        new_features = {}
        for dataset_meta in self.dataset_metadatas:
            if "human_image" in dataset_meta.features.keys():
                key_mapping = DATA_KEYS_MAPPING_HAND
                for visual_key in visual_keys_used:
                    new_features[visual_key] = dataset_meta.features.get(key_mapping[visual_key], None)
                break
        # check new_features has all visual_keys_used and none of them is None
        if not all(key in new_features and new_features[key] is not None for key in visual_keys_used):
            raise ValueError(f"Not all visual keys {visual_keys_used} found in the combined datasets' features or some are None.")
        # pop out all image and video features from dataset 0's features
        combined_info["features"] = {**new_features, **{k: v for k, v in self.dataset_metadatas[0].features.items() if (v['dtype'] != "image" and v['dtype'] != "video")}}
        
        # BUG:now the actions and state info is derived from human hand dataset, which may not be suitable for robot dataset

        # Aggregate counts
        for dataset_meta in self.dataset_metadatas:
            total_episodes += dataset_meta.total_episodes
            total_frames += dataset_meta.total_frames
            total_tasks += dataset_meta.total_tasks
            total_videos += len(dataset_meta.video_keys)
        
        combined_info.update({
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": total_videos,
        })
        
        return combined_info

    def _combine_tasks(self) -> tuple[dict, dict]:
        """Combine tasks from all datasets"""
        combined_tasks = {}
        combined_task_to_task_index = {}
        current_task_index = 0
        
        for dataset_meta in self.dataset_metadatas:
            for task_index, task in dataset_meta.tasks.items():
                if task not in combined_task_to_task_index:
                    combined_task_to_task_index[task] = current_task_index
                    combined_tasks[current_task_index] = task
                    current_task_index += 1
        
        return combined_tasks, combined_task_to_task_index

    def _combine_episodes(self) -> dict:
        """Combine episodes from all datasets"""
        combined_episodes = {}
        episode_offset = 0
        
        for dataset_meta in self.dataset_metadatas:
            for ep_index, episode in dataset_meta.episodes.items():
                # Adjust episode index to avoid conflicts
                adjusted_ep_index = ep_index + episode_offset
                combined_episodes[adjusted_ep_index] = episode.copy()
                combined_episodes[adjusted_ep_index]["original_dataset"] = dataset_meta.repo_id
                combined_episodes[adjusted_ep_index]["original_ep_index"] = ep_index
            
            episode_offset += dataset_meta.total_episodes
        
        return combined_episodes

    def _combine_stats(self) -> dict:
        """Combine stats from all datasets"""
        if not self.dataset_metadatas:
            raise ValueError("No dataset metadata available to combine stats.")

        # import ipdb; ipdb.set_trace()
        from lerobot.common.constants import ACTION, OBS_ROBOT, TRAJ, OBS_TRAJ
        needed_keys = [ACTION, OBS_ROBOT, TRAJ, OBS_TRAJ]
        combined_stats = {}
        state_traj_stats = aggregate_stats([{OBS_TRAJ:meta.stats[OBS_TRAJ].copy()} for meta in self.dataset_metadatas])
        traj_stats       = aggregate_stats([{TRAJ:meta.stats[TRAJ].copy()} for meta in self.dataset_metadatas])
        
        for dataset_meta in self.dataset_metadatas:
            if not 'human_image' in dataset_meta.stats.keys():
                action_stats = {ACTION:dataset_meta.stats[ACTION].copy()}
                state_action_stats = {OBS_ROBOT:dataset_meta.stats[OBS_ROBOT].copy()}
        
        combined_stats.update(state_traj_stats)
        combined_stats.update(traj_stats)
        combined_stats.update(action_stats)
        combined_stats.update(state_action_stats)

        # combined_stats = self.dataset_metadatas[0].stats.copy()
        
        # if len(self.dataset_metadatas) > 1:
        #     for dataset_meta in self.dataset_metadatas[1:]:
        #         if dataset_meta.stats:
        #             combined_stats = aggregate_stats([combined_stats, dataset_meta.stats])
        
        return combined_stats

    def _combine_episodes_stats(self) -> dict:
        """Combine episodes_stats from all datasets"""
        combined_episodes_stats = {}
        episode_offset = 0
        
        for dataset_meta in self.dataset_metadatas:
            for ep_index, ep_stats in dataset_meta.episodes_stats.items():
                adjusted_ep_index = ep_index + episode_offset
                combined_episodes_stats[adjusted_ep_index] = ep_stats
            
            episode_offset += dataset_meta.total_episodes
        
        return combined_episodes_stats

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        """Get data file path for a specific episode"""
        # Find which dataset this episode belongs to
        for dataset_meta in self.dataset_metadatas:
            if ep_index < dataset_meta.total_episodes:
                return dataset_meta.get_data_file_path(ep_index)
            ep_index -= dataset_meta.total_episodes
        raise ValueError(f"Episode index {ep_index} out of range")

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """Get video file path for a specific episode"""
        # Find which dataset this episode belongs to
        for dataset_meta in self.dataset_metadatas:
            if ep_index < dataset_meta.total_episodes:
                return dataset_meta.get_video_file_path(ep_index, vid_key)
            ep_index -= dataset_meta.total_episodes
        raise ValueError(f"Episode index {ep_index} out of range")

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get episode chunk for a specific episode"""
        # Find which dataset this episode belongs to
        for dataset_meta in self.dataset_metadatas:
            if ep_index < dataset_meta.total_episodes:
                return dataset_meta.get_episode_chunk(ep_index)
            ep_index -= dataset_meta.total_episodes
        raise ValueError(f"Episode index {ep_index} out of range")

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return sum(dataset_meta.total_chunks for dataset_meta in self.dataset_metadatas)

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return max(dataset_meta.chunks_size for dataset_meta in self.dataset_metadatas)

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        """Save episode to the combined dataset"""
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository IDs: {self.repo_ids},\n"
            f"    Number of datasets: {len(self.dataset_metadatas)},\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_ids: list[str],
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
    ) -> "LerobotCombinedDatasetMetadata":
        """Creates metadata for a combined LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_ids = repo_ids
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / "combined_dataset"

        obj.root.mkdir(parents=True, exist_ok=False)

        if robot is not None:
            features = get_features_from_robot(robot, use_videos)
            robot_type = robot.robot_type
        elif features is None:
            raise ValueError(
                "Dataset features must either come from a Robot or explicitly passed upon creation."
            )
        else:
            features = {**features, **DEFAULT_FEATURES}

            # check if none of the features contains a "/" in their names
            for key in features:
                if "/" in key:
                    raise ValueError(f"Feature names should not contain '/'. Found '/' in feature '{key}'.")

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        obj.info["repo_ids"] = repo_ids
        obj.info["num_datasets"] = len(repo_ids)
        
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        
        obj.revision = None
        return obj
    
class CollectiveDataloader(pl.LightningDataModule):
    def __init__(self, 
                 accelerator,
                 datasets, 
                 num_workers, 
                 batch_size, 
                 shuffle, 
                 sampler, 
                 device, 
                 drop_last=False,
                 mixing_mode="max_size_cycle",  # "sequential", "max_size_cycle", or "min_size"
                 dataset_weights=None):     # List of weights for each dataset
        super().__init__()
        self.mixing_mode = mixing_mode
        # If dataset_weights is None, use equal weights for all datasets
        # Otherwise, use the specified weights
        num_datasets = len(datasets.datasets)
        dataset_weights = dataset_weights if dataset_weights is not None else [1.0] * num_datasets
       
        normalized_weights = [w / sum(dataset_weights) for w in dataset_weights]
        self.dataset_weights = normalized_weights
        batch_size_list = [max(1, int(w * batch_size)) for w in normalized_weights]
        
        if accelerator.num_processes > 1:
            assert sampler is None, f'check if sampler is set already, such as episodes aware sampler in train.py'
        self.multiple_dataloaders = {}
        for i, (name, dataset) in enumerate(datasets.datasets.items()):
            print('process index : ', accelerator.process_index, ', ', 'num process:', accelerator.num_processes,  \
                  ',', "Length: ", len(dataset))
            if accelerator.num_processes > 1:
                sampler = DistributedSampler(dataset,
                                             num_replicas=accelerator.num_processes,
                                             rank=accelerator.process_index,
                                             shuffle=True)
                shuffle=False
            self.multiple_dataloaders[name] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size_list[i],
                shuffle=shuffle,
                num_workers=num_workers,
                sampler=sampler,
                pin_memory=device.type!="cpu",
                drop_last=drop_last,
            )
        
        # self.train_set = CollectiveDataset(
        #             datasets.datasets, 
        #             num_workers, 
        #             batch_size_list, 
        #             shuffle, 
        #             sampler, 
        #             device, 
        #             drop_last,
        #         ).datasets

    def train_dataloader(self):
        # Create a list of (dataset, weight) tuples for weighted sampling
        # dataset_values = list(self.train_set.values())
        return CombinedLoader(self.multiple_dataloaders, mode=self.mixing_mode)
        # return CombinedLoader(self.train_set, mode=self.mixing_mode)
       

# class CollectiveDataset:
#     def __init__(self, 
#                  datasets, 
#                  num_workers, 
#                  batch_size_list, 
#                  shuffle, 
#                  sampler, 
#                  device, 
#                  drop_last=False):
#         # datasets is a dictionary of {dataset_name : Dataset object}
#         loaded_dataloaders = {
#             name: torch.utils.data.DataLoader(
#                 dataset,
#                 batch_size=batch_size_list[i],  
#                 shuffle=shuffle,
#                 num_workers=num_workers,
#                 sampler=sampler,
#                 pin_memory=device.type != "cpu",
#                 drop_last=drop_last,
#             )
#             for i, (name, dataset) in enumerate(datasets.items())
#         }
#         self.datasets = loaded_dataloaders

class LerobotCombinedDataset: 
    def __init__(self, cfg, imagenet_stats, 
                 resolve_delta_timestamps_func, 
                 resolve_delta_traj_timestamps_func,
                 image_transforms, 
                 delta_transforms):
        self.repo_id = cfg.dataset.repo_id 
        self.cfg = cfg

        if not isinstance(self.repo_id, list):
            raise ValueError("repo_id must be a list")

        self.imagenet_stats = imagenet_stats
        self.resolve_delta_timestamps_func = resolve_delta_timestamps_func
        self.resolve_delta_traj_timestamps_func = resolve_delta_traj_timestamps_func
        self.image_transforms = image_transforms
        self.delta_transforms = delta_transforms
        
        self.datasets = {}
        for index, repo_id in enumerate(self.repo_id): 
            name = repo_id.split("/")[-1]
            
            ds_meta = LeRobotDatasetMetadata(
                repo_id, root=self.cfg.dataset.root, revision=self.cfg.dataset.revision
            )
            delta_timestamps = self.resolve_delta_timestamps_func(self.cfg.policy, ds_meta)
            delta_traj_timestamps = self.resolve_delta_traj_timestamps_func(self.cfg.policy, ds_meta, index)
            dataset = LeRobotDataset(
                repo_id,
                root=self.cfg.dataset.root,
                episodes=self.cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                delta_traj_timestamps=delta_traj_timestamps,
                image_transforms=self.image_transforms,
                delta_transforms=self.delta_transforms,
                normalize_option=self.cfg.dataset.normalize_option, 
                revision=self.cfg.dataset.revision,
                video_backend=self.cfg.dataset.video_backend,
                traj_sampling_fps=self.cfg.policy.traj_sampling_fps[index],
            )
            if self.cfg.dataset.use_imagenet_stats:
                # note the camera_keys
                for key in dataset.meta.camera_keys:
                    for stats_type, stats in self.imagenet_stats.items():
                        dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

            self.datasets[name] = dataset
        
    
    def __len__(self):
        return len(self.datasets)

    def init_meta(self): 
        # Initialize combined metadata covering all datasets
        self.meta = LerobotCombinedDatasetMetadata(
            repo_ids=self.repo_id,
            visual_keys_used=self.cfg.dataset.visual_keys_used,
            root=None,
            revision=self.cfg.dataset.revision,
            force_cache_sync=False,
        )

        # Note
        # self.meta.stats = None # stats is not needed for combined dataset
        # self.meta.episodes_stats = None # episodes_stats is not needed for combined dataset

        self.num_frames = self.meta.total_frames
        self.num_episodes = self.meta.total_episodes 
        # len(dataset.episode_data_index.keys()) == self.total_episodes
        self.episode_data_index = self.meta.episodes
        self.features = self.meta.features 
        self.total_tasks = self.meta.total_tasks
        self.total_chunks = self.meta.total_chunks
        self.chunks_size = self.meta.chunks_size