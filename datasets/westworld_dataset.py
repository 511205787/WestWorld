import bisect
import glob
import os

import h5py
import torch

from .base_dataset import BaseDataset


class WestWorldDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        self.use_kd = bool(getattr(config, "use_kd", False))
        kd_cfg = getattr(getattr(config, "method", None), "kd", None)
        self.kd_enabled = self.use_kd and bool(getattr(kd_cfg, "enabled", False))
        self.teacher_chunk_files = []
        self.teacher_chunk_counts = []

        if self.kd_enabled:
            teacher_h5_dir = None
            if kd_cfg is not None:
                if is_validation:
                    teacher_h5_dir = getattr(kd_cfg, "teacher_val_h5_dir", None)
                if teacher_h5_dir is None:
                    teacher_h5_dir = getattr(kd_cfg, "teacher_h5_dir", None)
            if not teacher_h5_dir:
                raise ValueError("KD enabled but teacher_h5_dir is not set.")

            teacher_h5_dir = os.path.expanduser(teacher_h5_dir)
            self.teacher_chunk_files = sorted(glob.glob(os.path.join(teacher_h5_dir, "chunk_*.h5")))
            if not self.teacher_chunk_files:
                raise FileNotFoundError(f"No teacher H5 chunks found under {teacher_h5_dir}")

            for f in self.teacher_chunk_files:
                with h5py.File(f, "r") as hf:
                    self.teacher_chunk_counts.append(hf["teacher_obs"].shape[0])
            if self.teacher_chunk_counts != self.lazy_chunk_counts:
                raise ValueError("Teacher H5 chunks do not align with base dataset chunks.")

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if not self.kd_enabled:
            return sample

        if self.valid_indices is not None:
            global_idx = self.valid_indices[idx]
        else:
            global_idx = idx

        chunk_idx = bisect.bisect_right(self.lazy_cum_counts, global_idx)
        local_idx = global_idx - (self.lazy_cum_counts[chunk_idx - 1] if chunk_idx > 0 else 0)

        with h5py.File(self.teacher_chunk_files[chunk_idx], "r") as hf:
            teacher_obs = torch.from_numpy(hf["teacher_obs"][local_idx])

        sample["teacher_obs"] = teacher_obs.float()
        return sample
