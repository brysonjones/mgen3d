
import sys
import absl.flags as flags
import yaml

from mgen3d.nerf.nerf import NeRF
from mgen3d.data.nerf_dataset import NeRFDataset
from mgen3d.nerf.utils import *


FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', "./config/default.yaml", 'Path to the config file')
flags.DEFINE_string('data_path', None, 'Path to the dataset root')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train on')

flags.mark_flag_as_required('data_path')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Pipeline:
    def __init__(self):
        with open(FLAGS.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        estimator = construct_estimator(self.config, device=device)
        network = construct_network(self.config, device=device)
        optimizer = construct_optimizer(self.config, network)
        scheduler = construct_scheduler(self.config, optimizer)
        self.nerf = NeRF(self.config, estimator, network, optimizer, scheduler)

    def run(self):    
        data_path = FLAGS.data_path
        train_dataset = NeRFDataset(data_path, "train", white_background=True, image_shape=(400, 400))
        eval_dataset = NeRFDataset(data_path, "val", white_background=True, image_shape=(400, 400))
        self.nerf.train(FLAGS.num_epochs, train_dataset, eval_dataset)

        
def main():
    flags.FLAGS(sys.argv)
    pipeline = Pipeline()
    pipeline.run()

if __name__ == '__main__':
    main()
