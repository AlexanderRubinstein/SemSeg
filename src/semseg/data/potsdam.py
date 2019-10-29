from os.path import join

import numpy as np
import os 

from .isprs import IsprsDataset
from .generators import FileGenerator, TRAIN, VALIDATION, TEST
from .utils import (
    save_img, load_img, get_img_size, compute_ndvi, _makedirs,
    save_numpy_array)

POTSDAM = 'potsdam'
PROCESSED_POTSDAM = 'processed_potsdam'

# dataset dependent parameters
class PotsdamDataset(IsprsDataset):
    sharah_train_ratio = 17 / 24

    def __init__(self, include_depth=True, include_ndvi=True):
        self.include_ir = True

        self.include_depth = include_depth
        self.include_ndvi = include_ndvi

        # DEBUG
        # For 3 active channels
        self.include_ir = False

        self.include_depth = False
        self.include_ndvi = False

        self.red_ind = 0
        self.green_ind = 1
        self.blue_ind = 2
        self.rgb_inds = [self.red_ind, self.green_ind, self.blue_ind]

        self.ir_ind = 3
        self.depth_ind = 4
        self.ndvi_ind = 5

        self.active_input_inds = list(self.rgb_inds)

        # Add extra channels to batch_x in addition to rgb
        if self.include_ir:
            self.active_input_inds.append(self.ir_ind)        
        if self.include_depth:
            self.active_input_inds.append(self.depth_ind)
        if self.include_ndvi:
            self.active_input_inds.append(self.ndvi_ind)


        super().__init__()

    def get_output_file_name(self, file_ind):
        return 'top_potsdam_{}_{}_label.tif'.format(file_ind[0], file_ind[1])

    def augment_channels(self, batch_x):
        red = batch_x[:, :, :, [self.red_ind]]
        ir = batch_x[:, :, :, [self.ir_ind]]
        ndvi = compute_ndvi(red, ir)
        return np.concatenate([batch_x, ndvi], axis=3)


class PotsdamFileGenerator(FileGenerator):
    """
    A data generator for the Potsdam dataset that creates batches from
    files on disk.
    """
#    def __init__(self, active_input_inds, train_ratio, cross_validation):
    def __init__(self, train_ratio):
        self.dataset = PotsdamDataset()

        # The first 24 indices correspond to the training set,
        # and the rest to the validation set used
        # in https://arxiv.org/abs/1606.02585
        self.file_inds = [
            (2, 10), (3, 10), (3, 11), (3, 12), (4, 11), (4, 12), (5, 10),
            (5, 12), (6, 10), (6, 11), (6, 12), (6, 8), (6, 9), (7, 11),
            (7, 12), (7, 7), (7, 9), (2, 11), (2, 12), (4, 10), (5, 11),
            (6, 7), (7, 10), (7, 8)
        ]

        self.test_file_inds = [
            (2, 13), (2, 14), (3, 13), (3, 14), (4, 13), (4, 14), (4, 15),
            (5, 13), (5, 14), (5, 15), (6, 13), (6, 14), (6, 15), (7, 13)
        ]

        # super().__init__(active_input_inds, train_ratio, cross_validation)
        super().__init__(self.dataset.active_input_inds, train_ratio, cross_validation=None)

class PotsdamImageFileGenerator(PotsdamFileGenerator):
    """
    A data generator for the Potsdam dataset that creates batches from
    the original TIFF and JPG files.
    """
    # def __init__(self, datasets_path, active_input_inds,
    #              train_ratio=0.8, cross_validation=None):
    #     self.dataset_path = join(datasets_path, POTSDAM)
    #     super().__init__(active_input_inds, train_ratio, cross_validation)

    def __init__(self, datasets_path,
                 train_ratio=0.8):
        self.dataset_path = join(datasets_path, POTSDAM)
        super().__init__(train_ratio)

    @staticmethod
    def preprocess(datasets_path):
        # Fix the depth image that is missing a column if it hasn't been
        # fixed already.
        data_path = join(datasets_path, POTSDAM)
        file_path = join(
            data_path,
            '1_DSM_normalisation/dsm_potsdam_03_13_normalized_lastools.jpg')

        im = load_img(file_path)
        if im.shape[1] == 5999:
            im_fix = np.zeros((6000, 6000), dtype=np.uint8)
            im_fix[:, 0:-1] = im[:, :, 0]
            save_img(im_fix, file_path)

    def get_file_size(self, file_ind):
        ind0, ind1 = file_ind

        rgbir_file_path = join(
            self.dataset_path,
            '4_Ortho_RGBIR' + os.sep + 'top_potsdam_{}_{}_RGBIR.tif'.format(ind0, ind1))
        nb_rows, nb_cols = get_img_size(rgbir_file_path)
        return nb_rows, nb_cols

    def get_img(self, file_ind, window, has_y=True):
        ind0, ind1 = file_ind

        rgbir_file_path = join(
            self.dataset_path,
            '4_Ortho_RGBIR' + os.sep + 'top_potsdam_{}_{}_RGBIR.tif'.format(ind0, ind1))
        depth_file_path = join(
            self.dataset_path,
            '1_DSM_normalisation' + os.sep + 'dsm_potsdam_{:0>2}_{:0>2}_normalized_lastools.jpg'.format(ind0, ind1)) # noqa
        batch_y_file_path = join(
            self.dataset_path,
            '5_Labels_for_participants' + os.sep + 'top_potsdam_{}_{}_label.tif'.format(ind0, ind1)) # noqa
        batch_y_no_boundary_file_path = join(
            self.dataset_path,
            '5_Labels_for_participants_no_Boundary' + os.sep + 'top_potsdam_{}_{}_label_noBoundary.tif'.format(ind0, ind1)) # noqa

        rgbir = load_img(rgbir_file_path, window)
        depth = load_img(depth_file_path, window)
        channels = [rgbir, depth]
        #DEBUG
        # print("rgbir.shape: {}".format(rgbir.shape))
        # print("depth.shape: {}".format(depth.shape))

        if has_y:
            batch_y = load_img(batch_y_file_path, window)
            # print("batch_y.shape: {}".format(batch_y.shape))
            batch_y_no_boundary = load_img(
                batch_y_no_boundary_file_path, window)
            # print("batch_y_no_boundary.shape: {}".format(batch_y_no_boundary.shape))
            channels.extend([batch_y, batch_y_no_boundary])

        img = np.concatenate(channels, axis=2)


        return img

    def parse_batch(self, batch, has_y=True):

        # DEBUG
        # print("Active input inds: {}".format(self.dataset.active_input_inds))
        # print("Batch shape with everything: {}".format(batch.shape))

        # DEBUG
        # batch_x = batch[:, :, :, 0:5]
        # Number of channels extracted from images
        nb_channels = len(self.dataset.active_input_inds) - 1
        # print("Number of channels without ndvi: {}".format(nb_channels))
        batch_x = batch[:, :, :, self.dataset.active_input_inds[0]:nb_channels]
        # print("Batch_x shape: {}".format(batch_x.shape))
        batch_y = None
        batch_y_mask = None
        if has_y:
            # batch_y = self.dataset.rgb_to_one_hot_batch(batch[:, :, :, 5:8])
            # batch_y_mask = self.dataset.rgb_to_mask_batch(batch[:, :, :, 8:])
            batch_y = self.dataset.rgb_to_one_hot_batch(batch[:, :, :, nb_channels:nb_channels+3])
            # print("Batch_y shape: {0} from {1}".format(batch_y.shape, batch[:, :, :, nb_channels:nb_channels+3].shape))

            # print("Batch_y_mask in rgb shape: {0}".format(batch[:, :, :, nb_channels+3:].shape))
            batch_y_mask = self.dataset.rgb_to_mask_batch(batch[:, :, :, nb_channels+3:])
            # print("Batch_y_mask shape: {0} and {1}".format(batch_y_mask.shape, batch[:, :, :, nb_channels+3:].shape))

        # batch_x = batch[:, :, :, self.active_input_inds[0]:self.active_input_inds[-1]]
        # batch_y = None
        # batch_y_mask = None
        # if has_y:
        #     batch_y = self.dataset.rgb_to_one_hot_batch(batch[:, :, :, self.active_input_inds[-1]:self.active_input_inds[-1] + 3])
        #     batch_y_mask = self.dataset.rgb_to_mask_batch(batch[:, :, :, self.active_input_inds[-1] + 3:])
        # DEBUG
        # print("Batch_x shape after parsing from image: {}".format(batch_x.shape))
        # if has_y:
        #     print("Batch_y shape after parsing from image: {}".format(batch_y.shape))
        #     print("Batch_y_mask shape after parsing from image: {}".format(batch_y_mask.shape))
        
        return batch_x, batch_y, batch_y_mask


class PotsdamNumpyFileGenerator(PotsdamFileGenerator):
    """
    A data generator for the Potsdam dataset that creates batches from
    numpy array files. This is about 20x faster than reading the raw files.
    """
    # def __init__(self, datasets_path, active_input_inds,
    #              train_ratio=0.8, cross_validation=None):
    #     self.raw_dataset_path = join(datasets_path, POTSDAM)
    #     self.dataset_path = join(datasets_path, PROCESSED_POTSDAM)
    #     super().__init__(active_input_inds, train_ratio, cross_validation)

    def __init__(self, datasets_path,
                 train_ratio=0.8):
        self.raw_dataset_path = join(datasets_path, POTSDAM)
        self.dataset_path = join(datasets_path, PROCESSED_POTSDAM)
        # super().__init__(active_input_inds, train_ratio, cross_validation) 
        if (train_ratio == None):
            train_ratio = 0.8    
        print("Train_ratio is: {0}".format(train_ratio))   
        super().__init__(train_ratio) 

    @staticmethod
    def preprocess(datasets_path):
        proc_data_path = join(datasets_path, PROCESSED_POTSDAM)
        _makedirs(proc_data_path)

        # generator = PotsdamImageFileGenerator(
        #     datasets_path, include_ir=True, include_depth=True,
        #     include_ndvi=False)
        generator = PotsdamImageFileGenerator(
            datasets_path
            )
        dataset = generator.dataset

        def _preprocess(split):
            gen = generator.make_split_generator(
                split, batch_size=1, shuffle=False, augment=False,
                normalize=False, eval_mode=True)

            # for batch_x, batch_y, batch_y_mask, file_inds in gen:

            for batch_tuple in gen: 
                batch_x = batch_tuple[0]
                batch_y = batch_tuple[1]
                batch_y_mask = batch_tuple[3]
                file_inds = batch_tuple[4]

                file_ind = file_inds[0]

                batch_x = np.squeeze(batch_x, axis=0)
                channels = [batch_x]


                if batch_y is not None:
                    batch_y = np.squeeze(batch_y, axis=0)
                    batch_y = dataset.one_hot_to_label_batch(batch_y)

                    # DEBUG
                    # print("Batch_y shape after squeezing and one_hot_batching is: {}".format(batch_y.shape))

                    batch_y_mask = np.squeeze(batch_y_mask, axis=0)
                    channels.extend([batch_y, batch_y_mask])
                channels = np.concatenate(channels, axis=2)

                # Indexes in name of produced .npy files
                ind0, ind1 = file_ind
                # ind0 = file_ind[0]
                # ind1 = file_ind[1]
                file_name = '{}_{}'.format(ind0, ind1)

                print("We are ready to save {0} to .npy and the batch shape is: {1}".format(file_name, channels.shape))

                # Creating numpy arrays from input images
                # DEBUG

                save_numpy_array(
                    join(proc_data_path, file_name), channels)

                # Free memory
                channels = None
                batch_x = None
                batch_y = None
                batch_y_mask = None

        _preprocess(TRAIN)
        _preprocess(VALIDATION)
        _preprocess(TEST)

        print ("Files have been preprocessed.")

    def get_file_path(self, file_ind):
        ind0, ind1 = file_ind
        return join(self.dataset_path, '{}_{}.npy'.format(ind0, ind1))

    def get_file_size(self, file_ind):
        file_path = self.get_file_path(file_ind)
        im = np.load(file_path, mmap_mode='r')
        nb_rows, nb_cols = im.shape[0:2]
        return nb_rows, nb_cols

    def get_img(self, file_ind, window, has_y=True):
        file_path = self.get_file_path(file_ind)
        im = np.load(file_path, mmap_mode='r')
        ((row_begin, row_end), (col_begin, col_end)) = window
        img = im[row_begin:row_end, col_begin:col_end, :]

        return img

    def parse_batch(self, batch, has_y=True):
        # batch_x = batch[:, :, :, 0:5]
        # print("Batch given to parse batch has shape: {}".format(batch.shape))
        nb_channels = self.dataset.active_input_inds[-1]
        # # DEBUG_BEGIN
        # nb_channels = len(self.active_input_inds)
        # print("Number of active channels: {0}".format(nb_channels))
        # # DEBUG_END
        batch_x = batch[:, :, :, self.dataset.active_input_inds[0]:nb_channels+1]
        batch_y = None
        batch_y_mask = None
        if has_y:

            # # DEBUG_BEGIN
            # test_batch_y_label = batch[:, :, :, nb_channels+1:nb_channels+2]
            # print("test_batch_y_label type: {0}".format(type(test_batch_y_label)))
            # print("test_batch_y_label shape: {0}".format(test_batch_y_label.shape))
            # print("batch with x, y and y_mask shape: {0}".format(batch.shape))
            # # print(test_batch_y_label)
            # # DEBUG_END

            # batch_y = self.dataset.label_to_one_hot_batch(batch[:, :, :, 5:6])
            batch_y = self.dataset.label_to_one_hot_batch(batch[:, :, :, -2:-1])
            # batch_y = self.dataset.label_to_one_hot_batch(batch[:, :, :, nb_channels+1:nb_channels+2])
            # batch_y_mask = batch[:, :, :, 6:7]
            batch_y_mask = batch[:, :, :, -1:]
            # batch_y_mask = batch[:, :, :, nb_channels+2:nb_channels+3]

            # # DEBUG_BEGIN
            # print("batch_y type: {0}".format(type(batch_y)))
            # print("batch_y shape: {0}".format(batch_y.shape))
            # # print(batch_y)
            # # DEBUG_END

            # DEBUG
            # print("Batch_y shape after parsing numpy array: {}".format(batch_y.shape))
            # print("Batch_y_mask shape after parsing numpy array: {}".format(batch_y_mask.shape))

        # DEBUG
        # print("Batch_x shape after parsing from numpy array: {}".format(batch_x.shape))


        return batch_x, batch_y, batch_y_mask
