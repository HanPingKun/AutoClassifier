import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import pkg_resources
import sklearn

from rxnfp.models import SmilesClassificationModel
logger = logging.getLogger(__name__)
model_path =  pkg_resources.resource_filename("rxnfp", "models/transformers/bert_mlm_1k_tpl")
print(model_path) 