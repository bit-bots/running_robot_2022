# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import sys
import gc
import os
from fractions import Fraction
from typing import Any, List
from dataclasses import dataclass
from functools import partial
from itertools import starmap

import torch
import numpy as np
import torchvision.transforms.functional as f
from pytorchvideo.data import UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy.lib.recfunctions import structured_to_unstructured

from nn_gaze_synthesis.utils.utils import get_closest

@dataclass
class Video:
    """
    Description of a video
    """

    uid: str
    path: str


class Ego4DDataset(torch.utils.data.Dataset):
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

        video_path = os.path.join(path, "full_scale")
        annotation_path = os.path.join(path, "gaze")

        videos = map(partial(os.path.join, video_path), os.listdir(annotation_path))
        videos = starmap(Video, enumerate(videos))
        videos = list(filter(lambda video: os.path.exists(video.path), videos))

        assert len(videos) > 0, "No videos loaded ☹️"

        self.encoded_videos = {
            v.uid: EncodedVideo.from_path(
                v.path, decode_audio=False
            )
            for v in tqdm(videos)
        }

        for v in tqdm(videos):
            self.clips.extend(
                list(get_all_clips(v, self.encoded_videos[v.uid].duration, self.sampler))
            )

        self.gaze_annotations = {}

        for v in tqdm(videos):
            annotation_file = os.path.join(annotation_path, os.path.basename(v.path))
            eye_data = np.genfromtxt(annotation_file, names=True, delimiter=",")
            eye_data = eye_data[["component_timestamp_s", "canonical_timestamp_s", "norm_pos_x", "norm_pos_y"]]
            self.gaze_annotations[v.uid] = eye_data

    def __len__(self):
        return len(self.clips)

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
        eye_data = self.gaze_annotations[video.uid]
        frame_times = np.linspace(clip_start, clip_end, frames.shape[0])  # Calculate frames
        sampled_eye_data = eye_data[get_closest(eye_data["canonical_timestamp_s"], frame_times)]
        sampled_eye_data = torch.from_numpy(structured_to_unstructured(sampled_eye_data[["norm_pos_x", "norm_pos_y"]]))

        # Flip axis, TODO evaluate
        sampled_eye_data[:, 1] = 1 - sampled_eye_data[:, 1]

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
