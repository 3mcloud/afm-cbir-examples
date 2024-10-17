import numpy as np
import scipy.linalg
from piblin.data import TwoDimensionalDataset
from piblin.transform import DatasetTransform
from piblin.transform import DynamicTransform
from piblin.transform import DatasetLambdaTransform


class ScipyTransform(DynamicTransform):
    _PACKAGE = "scipy"


class SubtractPlane(DatasetTransform):

    def _apply(self, target: TwoDimensionalDataset, **kwargs) -> TwoDimensionalDataset:

        image_data = target.dependent_variable_data

        # the image data needs to be reshaped to be (N x 3)
        x = np.arange(0, image_data.shape[0])
        y = np.arange(0, image_data.shape[1])

        nx = len(x)
        ny = len(y)
        xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
        data = []
        for i in range(nx):
            for j in range(ny):
                if not np.isnan(image_data[xv[i, j], yv[i, j]]):
                    data.append([xv[i, j], yv[i, j], image_data[xv[i, j], yv[i, j]]])

        data = np.array(data)

        # best-fit linear plane
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        # evaluate it on grid
        Z = C[0] * xv + C[1] * yv + C[2]

        target.dependent_variable_data = image_data - Z

        return target


def transform_function(dataset, **kwargs):
    dataset.dependent_variable_data = dataset.dependent_variable_data.astype(np.uint8)
    return dataset


cast_to_int = DatasetLambdaTransform(transform_function)


class Normalize(DatasetTransform):

    def _apply(self, target: TwoDimensionalDataset, **kwargs) -> TwoDimensionalDataset:
        """Apply this normalization to the dataset.

        Parameters
        ----------
        dataset
            The dataset to apply this normalization to.

        Returns
        -------
        dataset
            The normalized dataset after the application.
        """
        target_maximum = kwargs.get("target_maximum", 1.0)

        dataset_min_value = np.min(target.dependent_variable_data)
        dataset_max_value = np.max(target.dependent_variable_data)

        target.dependent_variable_data = \
            ((target.dependent_variable_data -
              dataset_min_value) / (dataset_max_value -
                                    dataset_min_value)) * target_maximum

        return target
