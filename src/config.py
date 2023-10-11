from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProjectConfig:
    seed: int=123
    
    data_dir_path: Path=Path("./input/")
    output_dir_path: Path=Path("./experiments/")
    models_dir_path: Path=Path("./models/")
    
    input_file_name: str="BTC_Tweets_Updated.csv"
    model_name_hf: str="decapoda-research/llama-7b-hf" #"meta-llama/Llama-2-7b-hf"
    #proba: str='proba'
    cutoff_len: int=256
    verbose: bool=True

# from argparse import ArgumentParser

# parser = ArgumentParser()

# def get_args():
#     parser.add_argument(
#         "--data_dir_path",
#         default="../input/",
#         help="File directory for input data."
#     )

#     parser.add_argument(
#         "--file_names",
#         help="File name or names to be imported. If more than one file enclose in a list. Concat index should be the same in all files."
#     )

#     return parser.parse_args()

