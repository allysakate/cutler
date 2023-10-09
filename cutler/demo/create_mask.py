import os
import time
import warnings

import multiprocessing as mp
from glob import glob
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import sys
sys.path.append('./')
sys.path.append('../')
from config import add_cutler_config, Parameters
from predictor import VisualizationDemo
warnings.filterwarnings('ignore')

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_cutler_config(cfg)
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Disable the use of SyncBN normalization when running on a CPU
    # SyncBN is not supported on CPU and can cause errors, so we switch to BN instead
    if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def process(config_data):
    param = Parameters(config_dict=config_data)
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Config Data: " + str(config_data))
    cfg = setup_cfg(param)
    demo = VisualizationDemo(cfg)

    img_files = []
    for img_ext in ["jpg", "png", "JPG"]:
        input_files = glob(os.path.join(param.input_dir, f"*.{img_ext}"))
        img_files.extend(input_files)
    logger.info(f"No. of images: {len(img_files)}")

    for img_file in img_files:
        # use PIL, to be consistent with evaluation
        img = read_image(img_file, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        pred_count = len(predictions["instances"])
        logger.info(f"Image: {img_file} has {pred_count}. Time: {time.time() - start_time}")

        # save image to your local directory
        if param.output_dir:
            if not os.path.exists(param.output_dir):
                os.makedirs(param.output_dir)
            if os.path.isdir(param.output_dir):
                assert os.path.isdir(param.output_dir), param.output_dir
                out_filename = os.path.join(param.output_dir, os.path.basename(img_file))
            visualized_output.save(out_filename)


if __name__ == "__main__":
    for pth_name in [
        "cutler_cascade_final",
        "cascade_mask_rcnn_R_50_FPN",
        "cascade_mask_rcnn_R_50_FPN_self_train_r1",
        "cascade_mask_rcnn_R_50_FPN_self_train_r2"
    ]:
        config_data = {
            "config_file": "../model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml",
            "confidence_threshold": 0.5,
            "opts": ["MODEL.WEIGHTS", f"{pth_name}.pth", "MODEL.DEVICE", "cuda:0"],
            "input_dir": "/home/allysakate/Videos/gensan/images/",
            "output_dir": f"/home/allysakate/Videos/cutler/gensan_test/{pth_name}",
        }
        process(config_data)
