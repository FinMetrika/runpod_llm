import sys
import logging
import random
import numpy as np
import torch
from config import ProjectConfig
from utils import update_config


# logging.basicConfig(filename="program_log.txt",
#                     level=logging.INFO,
#                     format='%(asctime)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')


#################################
def main():
    # Instantiate the config dataclass
    FLAGS = ProjectConfig()
    update_config(FLAGS)
    #logging.info(f"Configuration: {FLAGS}")

    # Random seeds
    seed = FLAGS.seed
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda
        



if __name__ == '__main__':
    #logging.info(f"Program started with arguments: {FLAGS}")
    main()