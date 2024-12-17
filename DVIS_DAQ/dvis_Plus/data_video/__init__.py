# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from .dataset_mapper_prw import PRWVISDatasetMapper
from .dataset_mapper import YTVISDatasetMapper, CocoClipDatasetMapper, YTVISPERSONDatasetMapper
from .dataset_mapper_vps import PanopticDatasetVideoMapper
from .dataset_mapper_vss import SemanticDatasetVideoMapper
from .build import *

from .datasets import *
from .prwvis_eval import PRWVISEvaluator
from .ytvis_eval import YTVISEvaluator
from .vps_eval import VPSEvaluator
from .vss_eval import VSSEvaluator
from .uninext_ytvis_eval import UniYTVISEvaluator

from .dataset_mapper_sot import SOTDatasetMapper
