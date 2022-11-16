import pytest
from embetter.vision import ImageLoader, ColorHistogramEncoder, TimmEncoder


@pytest.mark.parametrize("n_buckets", [5, 10, 25, 128])
def test_color_hist_resize(n_buckets):
    """Make sure we can resize and it fits"""
    X = ImageLoader().fit_transform(["tests/data/thiscatdoesnotexist.jpeg"])
    che = ColorHistogramEncoder(n_buckets=n_buckets)
    output = che.fit_transform(X)
    shape_out = output.shape
    shape_exp = (1, che.output_dim)
    assert shape_exp == shape_out
    assert len(che.get_feature_names_out()) == che.output_dim


@pytest.mark.parametrize("encode_predictions,size", [(True, 1000), (False, 1280)])
def test_basic_timm(encode_predictions, size):
    """Super basic check for torch image model."""
    model = TimmEncoder("mobilenetv2_120d", encode_predictions=encode_predictions)
    X = ImageLoader().fit_transform(["tests/data/thiscatdoesnotexist.jpeg"])
    out = model.fit_transform(X)
    assert out.shape == (1, size)
