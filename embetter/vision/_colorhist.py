import numpy as np
from embetter.base import EmbetterBase


class ColorHistogramEncoder(EmbetterBase):
    """
    Encoder that generates an embedding based on the color histogram of the image.

    ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/colorhistogram.png)

    Arguments:
        n_buckets: number of buckets per color

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline

    from embetter.grab import ColumnGrabber
    from embetter.vision import ImageLoader, ColorHistogramEncoder

    # Let's say we start we start with a csv file with filepaths
    data = {"filepaths":  ["tests/data/thiscatdoesnotexist.jpeg"]}
    df = pd.DataFrame(data)

    # Let's build a pipeline that grabs the column, turns it
    # into an image and embeds it.
    pipe = make_pipeline(
        ColumnGrabber("filepaths"),
        ImageLoader(),
        ColorHistogramEncoder()
    )

    # This pipeline can now encode each image in the dataframe
    pipe.fit_transform(df)
    ```
    """

    def __init__(self, n_buckets=256):
        self.n_buckets = n_buckets
        self.output_dim = self.n_buckets * 3

    def transform(self, X, y=None):
        """
        Takes a sequence of `PIL.Image` and returns a numpy array representing
        a color histogram for each.
        """
        output = np.zeros((len(X), self.output_dim))
        for i, x in enumerate(X):
            arr = np.array(x)
            output[i, :] = np.concatenate(
                [
                    np.histogram(
                        arr[:, :, 0].flatten(),
                        bins=np.linspace(0, 255, self.n_buckets + 1),
                    )[0],
                    np.histogram(
                        arr[:, :, 1].flatten(),
                        bins=np.linspace(0, 255, self.n_buckets + 1),
                    )[0],
                    np.histogram(
                        arr[:, :, 2].flatten(),
                        bins=np.linspace(0, 255, self.n_buckets + 1),
                    )[0],
                ]
            )
        return output

    def get_feature_names_out(self, feature_names_out=None):
        return [f"colorhist_{i}" for i in range(self.output_dim)]
