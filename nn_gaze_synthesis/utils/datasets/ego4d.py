# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import os
from fractions import Fraction
from typing import Any, List
from dataclasses import dataclass

import torch
from pytorchvideo.data import UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader



@dataclass
class Video:
    """
    Description of a video
    """

    uid: str
    path: str


class IndexableVideoDataset(torch.utils.data.Dataset):
    def __init__(self, videos, sampler, transform):
        self.clips = []
        self.sampler = sampler
        self.transform = transform

        self.encoded_videos = {
            v.uid: EncodedVideo.from_path(
                v.path, decode_audio=False, perform_seek=False
            )
            for v in videos
        }

        for v in videos:
            self.clips.extend(
                list(get_all_clips(v, self.encoded_videos[v.uid].duration, sampler))
            )

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

        frames = self.encoded_videos[video.uid].get_clip(clip_start, clip_end)["video"]

        sample_dict = {
            "video": frames,
            "video_name": video.uid,
            "video_index": idx,
            "clip_index": clip_index,
            "aug_index": aug_index
        }
        sample_dict = self.transform(sample_dict)
        return sample_dict


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


def create_dset(path, sequence_length=20, fps=10) -> IndexableVideoDataset:
    clip_sampler = UniformClipSampler(
        clip_duration=Fraction(
            sequence_length, fps
        ),
        backpad_last=True,
    )

    video_path = os.path.join(path, "full_scale")
    annotation_path = os.path.join(path, "full_scale")

    videos = [Video(i, os.path.join(video_path, name)) for i, name in enumerate(os.listdir(annotation_path))]
    videos = list(filter(os.path.exists, videos))

    assert len(videos) > 0, "No videos loaded :("

    transform = None  # TODO
    return IndexableVideoDataset(videos, clip_sampler, transform)
