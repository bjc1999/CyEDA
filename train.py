import time
import copy
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()

testopt = copy.deepcopy(opt)
testopt.nThreads = 1   # test code only supports nThreads = 1
testopt.batchSize = 1  # test code only supports batchSize = 1
testopt.no_flip = True  # no flip
testopt.phase = 'val'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

test_data_loader = CreateDataLoader(testopt)
test_dataset = test_data_loader.load_data()
test_dataset_size = len(test_data_loader)

print('#training images = %d' % dataset_size)
print('#testing images = %d' % test_dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    model.train()

    if opt.continue_train and not opt.which_epoch == 'latest':
        epoch = epoch + int(opt.which_epoch)
        total_steps = dataset_size * (epoch - 1)

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch, n_cols=3, display_size=opt.fineSize)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.eval()
    test_results = []
    test_details = []
    for i, test_data in enumerate(test_dataset):
        model.set_input(test_data)
        result = model.predict(i)
        test_results.extend(result)
    visualizer.display_current_results(OrderedDict(test_results), epoch, phase='val', n_cols=8, display_size=opt.fineSize)

    if epoch > opt.niter:
        model.update_learning_rate()
