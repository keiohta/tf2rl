import os
import argparse
from glob import glob

from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util


def get_average_test_return(path):
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag == "Common/average_test_return":
                t = tensor_util.MakeNdarray(v.tensor)
                print(e.step, v.simple_value, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='results')
    parser.add_argument('--env-name', type=str, default="HalfCheetah-v2")
    args = parser.parse_args()

    directories = glob(os.path.join(args.root_dir, "*"))

    for dir in directories:
        if args.env_name in dir:
            print(dir)

    path = "results/20191203T204944.997591_DDPG_InvertedDoublePendulum-v2/events.out.tfevents.1575373785.KeinoMac-mini.local.97248.353.v2"
    get_average_test_return(path)
