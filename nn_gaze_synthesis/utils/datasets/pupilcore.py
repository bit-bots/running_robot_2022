# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
from abc import abstractmethod
import gc
import os
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from itertools import starmap
from typing import Any, List

import msgpack
import numpy as np
import torch
import torchvision.transforms.functional as f
from nn_gaze_synthesis.utils.utils import get_closest
from numpy.lib import recfunctions as rfn
from numpy.lib.recfunctions import structured_to_unstructured
from pytorchvideo.data import UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class Video:
    """
    Description of a video
    """

    uid: str
    path: str


class _FrozenDict(dict):
    def __setitem__(self, key, value):
        raise NotImplementedError('Invalid operation')

    def clear(self):
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        raise NotImplementedError()


class PupilcoreTrackerDataset(torch.utils.data.Dataset):
    def __init__(self, path, sampler=None, transform=None, sequence_length=40, fps=15):
        self.fps = fps
        self.sequence_length = sequence_length
        self.clips = []
        self.transform = transform

        if sampler is None:
            self.sampler = UniformClipSampler(
                clip_duration=Fraction(
                    sequence_length, fps
                    ),
                    backpad_last=True,
                )
        else:
            self.sampler = sampler

        trials = list(map(partial(os.path.join, path), os.listdir(path)))

        print(trials)

        videos = list(starmap(Video, enumerate(trials)))

        print(videos)

        self.encoded_videos = {
            v.uid: EncodedVideo.from_path(
                os.path.join(v.path, 'world.mp4'), decode_audio=False
            )
            for v in tqdm(videos)
        }

        for v in tqdm(videos):
            self.clips.extend(
                list(get_all_clips(v, self.encoded_videos[v.uid].duration, self.sampler))
            )

        self.gaze_annotations = {}

        for v in tqdm(videos):
            annotation_file = os.path.join(v.path, "gaze.pldata")
            gaze_info = PupilcoreTrackerDataset._load_gaze_file(annotation_file)
            gaze_db = PupilcoreTrackerDataset._get_interpolating_gaze_db(gaze_info)
            self.gaze_annotations[v.uid] = gaze_db

    def __len__(self):
        return len(self.clips)

    @staticmethod
    def _unpacking_object_hook(obj):
        if type(obj) is dict:
            return _FrozenDict(obj)

    @staticmethod
    def _load_gaze_file(file):
        with open(file, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            return [msgpack.unpackb(
                buffer,
                raw=False,
                use_list=False,
                object_hook=PupilcoreTrackerDataset._unpacking_object_hook) for name, buffer in unpacker if name == 'gaze.3d.01.']

    @staticmethod
    def _get_interpolating_gaze_db(gaze_info):
        # Extract used data
        gaze_timestamps = np.fromiter(
            (info['timestamp'] for info in gaze_info), float)
        gaze_points = rfn.structured_to_unstructured(np.fromiter(
            (info['norm_pos'] for info in gaze_info), dtype=('f4,f4')))

        # Create interpolation object
        return interp1d(gaze_timestamps, gaze_points, axis=0, bounds_error=False, fill_value="extrapolate")

    def __getitem__(self, idx):
        video, clip = self.clips[idx]

        (
            clip_start,
            clip_end,
            clip_index,
            aug_index,
            _,
        ) = clip
        
        gc.collect()

        frames = uniform_temporal_subsample(self.encoded_videos[video.uid].get_clip(clip_start, clip_end)["video"], self.sequence_length).transpose(0, 1)

        # Load eye track data
        gaze_db = self.gaze_annotations[video.uid]
        frame_times = np.linspace(clip_start, clip_end, frames.shape[0])  # Calculate frames
        world_frame_timestamps = np.load(os.path.join(video.path, "world_timestamps.npy"))
        frame_times += world_frame_timestamps[0]
        # Interpolate
        sampled_eye_data = torch.from_numpy(gaze_db(frame_times))

        # Flip axis
        # sampled_eye_data[:, 1] = 1 - sampled_eye_data[:, 1]

        sample = (f.resize(frames, (224, 224)).float() / 255, sampled_eye_data.float())

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def get_all_clips(video, video_length, sampler):
    last_clip_time = None
    annotation = {}
    n_clips = 0
    while True:
        clip = sampler(last_clip_time, video_length, annotation)
        last_clip_time = clip.clip_end_sec
        n_clips += 1

        yield (video, clip)

        if clip.is_last_clip:
            break