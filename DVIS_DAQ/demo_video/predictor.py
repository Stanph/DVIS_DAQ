# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# reference: https://github.com/sukjunhwang/IFC/blob/master/projects/IFC/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from visualizer import TrackVisualizer

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.modeling import build_model
import detectron2.data.transforms as T
import numpy as np

def _get_objects_from_outputs(outputs):
    def _get_objects_from_vis_outputs(outputs):
        pred_scores = outputs["pred_scores"]
        pred_labels = outputs["pred_labels"]
        pred_masks = outputs["pred_masks"]
        if 'pred_ids' in outputs.keys():
            pred_ids = outputs['pred_ids']
        else:
            pred_ids = None
            pred_ids_ = None

        # filter low score instance prediction
        pred_scores_ = []
        pred_labels_ = []
        pred_masks_ = []
        if pred_ids is not None:
            pred_ids_ = []
        for i, score in enumerate(pred_scores):
            if score < 0.01:
                continue
            pred_scores_.append(pred_scores[i])
            pred_labels_.append(pred_labels[i])
            pred_masks_.append(pred_masks[i])
            if pred_ids is not None:
                pred_ids_.append(pred_ids[i])

        return pred_masks_, pred_labels_, pred_scores_, pred_ids_

    def _get_objects_from_vss_outputs(outputs):
        # init
        pred_labels = []
        pred_masks = []
        pred_scores = []
        pred_ids = []

        sem_seg = outputs['pred_masks']
        sem_cats = np.unique(sem_seg)  # get valid class
        for cls in sem_cats:
            pred_scores.append(1)
            pred_labels.append(cls)
            pred_masks.append(sem_seg == cls)
            pred_ids.append(cls)  # using class ID as object ID
        return pred_masks, pred_labels, pred_scores, pred_ids

    def _get_objects_from_vps_outputs(outputs):
        segments_infos = outputs['segments_infos']
        pred_ids = outputs['pred_ids']

        # generate object score list
        pred_scores = [1 for segments_info in segments_infos]
        outputs["pred_scores"] = pred_scores

        # get bit-mask and category lable for per object(thing & stuff)
        pred_labels = []
        pred_masks = []
        pan_seg = outputs['pred_masks']
        for segments_info in segments_infos:
            id = segments_info['id']
            pred_masks.append(pan_seg == id)
            pred_labels.append(segments_info['category_id'])
        return pred_masks, pred_labels, pred_scores, pred_ids

    func_dict = {
        'vis': _get_objects_from_vis_outputs,
        'vss': _get_objects_from_vss_outputs,
        'vps': _get_objects_from_vps_outputs
    }

    if 'task' in outputs.keys():
        pred_masks, pred_labels, pred_scores, pred_ids = func_dict[outputs['task']](outputs)
    else:
        # minvis prediction results
        pred_masks, pred_labels, pred_scores, pred_ids = _get_objects_from_vis_outputs(outputs)

    return pred_masks, pred_labels, pred_scores, pred_ids

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = VideoPredictor(cfg)
        self.id_memories = {}

    def run_on_video(self, frames):
        """
        Args:
            frames (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(frames)

        image_size = predictions["image_size"]
        pred_masks, pred_labels, pred_scores, pred_ids = _get_objects_from_outputs(predictions)

        frame_masks = list(zip(*pred_masks))
        total_vis_output = []
        for frame_idx in range(len(frames)):
            frame = frames[frame_idx][:, :, ::-1]
            visualizer = TrackVisualizer(frame, self.metadata, instance_mode=self.instance_mode)
            ins = Instances(image_size)
            if len(pred_scores) > 0:
                ins.scores = pred_scores
                ins.pred_classes = pred_labels
                ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)

            vis_output = visualizer.draw_instance_predictions(predictions=ins)
            total_vis_output.append(vis_output)

        return predictions, total_vis_output

class VisualizationDemo_windows(VisualizationDemo):
    def run_on_video(self, frames, keep=False):
        """
        Args:
            frames (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor((frames, keep))

        pred_masks, pred_labels, pred_scores, pred_ids = _get_objects_from_outputs(predictions)
        image_size = predictions["image_size"]

        frame_masks = list(zip(*pred_masks))
        total_vis_output = []
        for frame_idx in range(len(frames)):
            frame = frames[frame_idx][:, :, ::-1]
            visualizer = TrackVisualizer(frame, self.metadata,
                                         instance_mode=self.instance_mode,
                                         id_memories=self.id_memories)
            ins = Instances(image_size)
            if len(pred_scores) > 0:
                ins.scores = pred_scores
                ins.pred_classes = pred_labels
                ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)

            vis_output = visualizer.draw_instance_predictions(predictions=ins, ids=pred_ids)
            total_vis_output.append(vis_output)

        return predictions, total_vis_output

class VideoPredictor(DefaultPredictor):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        weight = torch.load(cfg.MODEL.WEIGHTS)
        if 'model' in weight.keys():
            weight = weight['model']
        self.model.load_state_dict(weight, strict=True)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, frames):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        # add for dvis processes long video
        if isinstance(frames, tuple):
            frames, keep = frames
        else:
            keep = False
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            input_frames = []
            for original_image in frames:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input_frames.append(image)

            inputs = {"image": input_frames, "height": height, "width": width, "keep": keep}
            predictions = self.model([inputs])
            return predictions


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = VideoPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
