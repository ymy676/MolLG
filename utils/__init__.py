
import argparse
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config yaml"
    )

    return parser.parse_args()


    
def create_experiment_dir(cfg):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_name = f"{cfg.experiment_name}_{timestamp}"

    exp_dir = os.path.join(
        cfg.save.output_dir,
        exp_name
    )

    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir
