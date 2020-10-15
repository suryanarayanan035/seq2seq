import tensorflow as tf 
from model.model import create_model
from model.input_fn import train_input_fn
from model.loss import composite_loss
from model.utils import Vocabulary
import os,json

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
STATIC_CONFIG = dict(json.load(open('config.json','r')))
print(staticmethod)
RUNTIME_CONFIG = {"root_path":ROOT_PATH}
print(RUNTIME_CONFIG)
CONFIG = {**STATIC_CONFIG,**RUNTIME_CONFIG}

vocabulary = Vocabulary()

next_training_batch = train_input_fn(vocabulary,CONFIG)

modeloutput - create_model(next_training_batch,CONFIG,True)
