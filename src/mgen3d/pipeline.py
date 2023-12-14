
import sys
import absl.flags as flags


FLAGS = flags.FLAGS
flags.DEFINE_string('image_path', None, 'Path to the image file')

flags.mark_flag_as_required('image_path')

class Pipeline:
    def __init__(self):
        self.nerf = None
        self.diffusion = None
        self.clip = None
        

    def run(self):
        pass

def main():
    image_path = FLAGS.image_path

if __name__ == '__main__':
    flags.FLAGS(sys.argv)
    main()
