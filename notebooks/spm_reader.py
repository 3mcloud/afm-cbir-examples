from typing import Union
import re
import numpy as np
import pySPM
from piblin.dataio import FileReader
from piblin.data import Measurement, MeasurementSet, TwoDimensionalDataset
from piblin.dataio.fileio.read import FileParsingException


class SpmReader(FileReader):

    supported_extensions = {"spm"}
    """The extension supported by this file reader class"""

    @property
    def default_mode(self) -> str:
        return ""

    def _read_file_contents(self, filepath: str,
                            **read_kwargs) -> pySPM.Bruker:
        """Read the contents of an .spm file into a pyspm object.

        Parameters
        ----------
        filepath
            The path to the file to read.

        Returns
        -------
        The contents read from the file.
        """
        return pySPM.Bruker(filepath)

    @staticmethod
    def _parse_str(value: str) -> Union[int, float, str]:
        """Convert a string to a numerical value if possible.

        Parameters
        ----------
        value : str
            The value to convert.

        Returns
        -------
        int or float or str
            The value (converted or not).
        """
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    @classmethod
    def _data_from_file_contents(cls,
                                 file_contents: pySPM.Bruker,
                                 file_location=None,
                                 file_name=None,
                                 **read_kwargs) -> MeasurementSet:

        channels_to_read = read_kwargs.get("channels_to_read", None)
        normalized_channels_to_read = set()
        if channels_to_read is not None:
            for channel in channels_to_read:
                normalized_channels_to_read.add(channel.lower())

        datasets = []
        details = {}

        for i, layer in enumerate(file_contents.layers):

            try:
                sub_names = [key for key in layer.keys() if
                             'image data' in str(key).lower()]
                sub_name = sub_names[0]
                layer_name = layer[sub_name][0].decode('latin1')

            except KeyError:
                raise FileParsingException(
                    f"Could not locate name of layer {i}")

            channel_name = re.match(
                r'([^ ]+) \[([^\]]*)\] "([^"]*)"',
                layer_name
            ).groups()[-1]

            if channels_to_read is not None:
                if channel_name.lower() not in normalized_channels_to_read:
                    continue

            for key, value in layer.items():
                if key != sub_name:  # b'@2:Image Data':
                    if len(value) == 1:

                        normalized_channel_name = \
                            channel_name.lower().replace(" ", "_")

                        normalized_detail_name = \
                            key.decode('latin1').lower().replace(" ", "_")

                        detail_name = f"{normalized_channel_name}_{normalized_detail_name}"

                        detail_value = \
                            SpmReader._parse_str(value[0].decode('latin1'))

                        details[detail_name] = detail_value
                    else:
                        print(f"Warning: Could not parse metadata "
                              f"{key.decode('latin1')}")

            image: pySPM.SPM.SPM_image = \
                file_contents.get_channel(channel_name)

            details["type:"] = image.type
            details["direction:"] = image.direction

            x_values = np.linspace(0, image.size["real"]["x"],
                                   image.size["pixels"]["x"])
            y_values = np.linspace(0, image.size["real"]["y"],
                                   image.size["pixels"]["y"])

            independent_variable_unit = image.size["real"]["unit"]

            if image.zscale == "?":
                dependent_variable_unit = None
            else:
                dependent_variable_unit = image.zscale

            datasets.append(TwoDimensionalDataset(
                dependent_variable_data=image.pixels.T,
                dependent_variable_names=[image.channel],
                dependent_variable_units=[dependent_variable_unit],
                independent_variable_data=[x_values, y_values],
                independent_variable_names=["x", "y"],
                independent_variable_units=[independent_variable_unit,
                                            independent_variable_unit]))

        measurement_ = Measurement(datasets=datasets, details=details)

        return MeasurementSet(measurements=[measurement_])
