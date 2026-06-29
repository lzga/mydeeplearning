# -*- coding: utf-8 -*-
"""
Compatibility helpers for old torchvision.datapoints API and newer torchvision.tv_tensors API.

This file only bridges API names. It does not change the dataset or transform logic.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torchvision

try:
    # Old torchvision API, e.g. torchvision 0.15.x
    from torchvision import datapoints  # type: ignore
    USING_OLD_DATAPOINTS = True
except ImportError:
    # New torchvision API, e.g. torchvision 0.20+
    from torchvision import tv_tensors

    USING_OLD_DATAPOINTS = False

    def _get_bbox_format(name: str):
        if hasattr(tv_tensors, "BoundingBoxFormat"):
            return getattr(tv_tensors.BoundingBoxFormat, name)
        return name

    class _BoundingBoxFormat:
        XYXY = _get_bbox_format("XYXY")
        XYWH = _get_bbox_format("XYWH")
        CXCYWH = _get_bbox_format("CXCYWH")

    class _CompatBoundingBox(tv_tensors.BoundingBoxes):
        """
        A shared BoundingBoxes subclass that accepts the old constructor keyword
        spatial_size=(H, W), while internally using the new canvas_size=(H, W).

        It is imported by both coco_dataset.py and transforms.py, so transform
        dispatch sees exactly the same class type.
        """

        def __new__(cls, data, format=None, spatial_size=None, canvas_size=None, **kwargs):
            if canvas_size is None:
                canvas_size = spatial_size
            if canvas_size is None:
                raise ValueError("BoundingBox requires spatial_size or canvas_size")
            return super().__new__(cls, data, format=format, canvas_size=tuple(canvas_size), **kwargs)

        @property
        def spatial_size(self):
            # Old API name used by this RT-DETR project.
            return tuple(self.canvas_size)

    datapoints = SimpleNamespace(
        Image=tv_tensors.Image,
        Video=getattr(tv_tensors, "Video", torch.Tensor),
        Mask=tv_tensors.Mask,
        BoundingBox=_CompatBoundingBox,
        BoundingBoxFormat=_BoundingBoxFormat,
    )
