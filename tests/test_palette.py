import pytest

from brickcell.palette import load_palette, parse_palette


def test_palette_has_12_main_colours_and_more():
    p = load_palette()
    assert len(p.names(main_only=True)) == 12
    assert len(p.names()) > 12
    for name in ("red", "azure", "yellow", "white", "brown"):  # seen at the station
        assert p.colors[name].seen


def test_nearest_names_station_colours():
    p = load_palette()
    assert p.nearest([0.702, 0.165, 0.155])[0] == "red"
    assert p.nearest([0.141, 0.667, 0.749])[0] == "azure"
    assert p.nearest([0.753, 0.569, 0.204])[0] == "yellow"
    assert p.nearest([0.0, 0.31, 0.70])[0] == "blue"
    assert p.nearest([0.05, 0.05, 0.06])[0] == "black"


def test_bad_palette():
    with pytest.raises(ValueError):
        parse_palette({"colors": {"x": {"albedo": [2, 0, 0]}}})
    with pytest.raises(ValueError):
        parse_palette({"colors": {}})
