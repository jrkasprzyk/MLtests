from pathlib import Path
import argparse
from typing import Dict

def set_supervisor_path(computer) -> Path:
    if computer == 'hpc':
        return Path("/projects/joka0958/supervisor/")
    elif computer == '2020laptop':
        return Path("C:/Users/josep/OneDrive - UCB-O365/Students/_shares/Lee HUB/junresearch/DeeplearningCNN_flow_detection/supervisor")
    elif computer == 'seecdesktop':
        return Path("C:/Users/joka0958/OneDrive - UCB-O365/Datasets/flow_detection/")

def set_output_path(computer) -> Path:
    if computer == 'hpc':
        return Path("/projects/joka0958/MLoutput/")
    elif computer == '2020laptop':
        return Path("C:/Users/josep/Documents/GitHub/MLtests/flow_detection/")
    elif computer == 'seecdesktop':
        return Path("//files.colorado.edu/CEAE/users/joka0958/Documents/GitHub/MLtests/flow_detection/")
