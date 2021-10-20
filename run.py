from deconv import REGISTRY
from tifffile import imread
from data_augmentation import DataAugmenter


def run(config):
    runner = REGISTRY[config['algorithm']](config)

    data_augmenter = DataAugmenter(config)
    (augmented_raw, augmented_gt, axes), data_dir = data_augmenter.augment(config['data_path'], config['source_folder'],
                                                                           config['target_folder'], care=True,
                                                                           data_dir=runner.res_path)

    runner.preprocess()
    model_dir, mdl_path = runner.train(data_dir, config['validation_split'],config['epochs'], config['batch_size'],
                                       config['train_steps'])

    y_test = imread(config['test_file_y'])
    x_test = imread(config['test_file_x'])
    runner.predict(x_test, model_dir, Y= y_test, plot =True)

    return 0