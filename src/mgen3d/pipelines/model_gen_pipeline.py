
import sys
import absl.flags as flags
from mgen3d.nerf.nerf import NeRF


FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', "./config/default.yaml", 'Path to the config file')

flags.mark_flag_as_required('config_path')

class Pipeline:
    def __init__(self, config):
        self.nerf = None
        self.stable_diffusion = None
        self.depth_model = None
        
    def run(self):
        pass

def main():
    flags.FLAGS(sys.argv)
    pipeline = Pipeline()
    pipeline.run()


if __name__ == '__main__':
    flags.FLAGS(sys.argv)
    main()
