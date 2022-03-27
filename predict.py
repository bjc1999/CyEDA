from collections import OrderedDict
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

print(len(dataset))
test_results = []
test_details = []
for i, data in enumerate(dataset):
    model.set_input(data)
    result = model.predict(i)
    img_path = model.get_image_paths()[0]
    print('process image... %s' % img_path)
    visualizer.save_images(OrderedDict(result), img_path)
