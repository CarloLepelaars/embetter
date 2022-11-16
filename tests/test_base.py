import pandas as pd
from embetter.grab import ColumnGrabber


def test_grab_column():
    """Ensure that we can grab a text column."""
    data = [{"text": "hi", "foo": 1}, {"text": "yes", "foo": 2}]
    dataframe = pd.DataFrame(data)
    out = ColumnGrabber("text").fit_transform(dataframe)
    assert out == ["hi", "yes"]


def test_grab_get_feature_names_out():
    colname = "test_col"
    cg = ColumnGrabber(colname=colname)
    assert cg.get_feature_names_out() == [colname]
