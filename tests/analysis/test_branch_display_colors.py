from roc_mcs.analysis.branch_display import normalize_branch_colors


def test_normalize_branch_colors_repairs_duplicate_legacy_overrides_deterministically():
    branch_ids = [7, 2, 5, 2]
    legacy_overrides = {
        "2": "#ff0000",
        "5": "#FF0000",
        "7": "not-a-color",
        "99": "#00FF00",
    }

    first = normalize_branch_colors(branch_ids, legacy_overrides)
    second = normalize_branch_colors(branch_ids, legacy_overrides)

    assert first == second
    assert set(first) == {2, 5, 7}
    assert first[2] == "#FF0000"  # Lowest ID keeps the valid saved override.
    assert len(set(first.values())) == len(first)
    assert all(len(value) == 7 and value.startswith("#") for value in first.values())


def test_normalize_branch_colors_keeps_distinct_saved_overrides():
    colors = normalize_branch_colors(
        [3, 1, 2],
        {1: "#112233", 2: "#445566", 3: "#778899"},
    )

    assert colors == {1: "#112233", 2: "#445566", 3: "#778899"}


def test_generated_defaults_do_not_take_colors_reserved_by_later_saved_overrides():
    from roc_mcs.analysis.branch_display import default_branch_color

    reserved = default_branch_color(1)
    colors = normalize_branch_colors([1, 2], {2: reserved})

    assert colors[2] == reserved
    assert colors[1] != reserved
    assert len(set(colors.values())) == 2
