import os
import ntpath
import time
from . import util
from PIL import Image
import math

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = True
        self.win_size = opt.display_winsize
        self.name = opt.name

        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, epoch, phase='train', color_channel='RGB', n_cols=4, display_size=200):
        self.saved = True
        extension = 'jpg' if not color_channel == 'LAB' else 'tiff'
        save_fname = f'epoch{epoch:03d}_{phase}.{extension}'
        display_width = list(visuals.items())[0][1].shape[1]
        max_width = display_width * n_cols
        max_height = display_size * math.ceil(len(visuals) / n_cols)
        save_im = Image.new(color_channel, (max_width, max_height))
        idx = 1
        width = 0
        height = 0
        # save images to the disk
        for label, image in visuals.items():                
            pil_im = Image.fromarray(image, color_channel).resize((display_width, display_size))
            save_im.paste(pil_im, (width, height))
            if idx%n_cols == 0:
                width = 0
                height += display_size
            else:
                width += display_width
            idx += 1
        img_path = os.path.join(self.img_dir, save_fname)
        save_im.save(img_path)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    
    # save image to the disk
    def save_images(self, visuals, image_path):
        short_path = ntpath.basename(image_path)
        name = os.path.splitext(short_path)[0]

        for label, image_numpy in visuals.items():
            if label.startswith('fake_B'):
                image_name = '%s.png' % (name)
                save_path = os.path.join('../bdd100k/valAB', image_name)
                util.save_image(image_numpy, save_path)

