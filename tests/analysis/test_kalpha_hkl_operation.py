from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roc_mcs.analysis.artifacts import ArtifactBundle, ArtifactStore
from roc_mcs.analysis.operations import _validate_kalpha_hkl, default_registry
from roc_mcs.processing.kalpha_identification import (
    KAlphaIdentificationConfig, identify_kalpha_hkl,
)
from roc_mcs.theory.bragg import BraggDoubletConfig, calculate_doublet_reflections
from roc_mcs.theory.crystals import CA_F2_FLUORITE


def _catalog_for_reflection(hkl="(004)", n=7, *, reused_partner=False):
    reflections, _ = calculate_doublet_reflections(CA_F2_FLUORITE, BraggDoubletConfig())
    theory = float(reflections.loc[reflections["hkl"] == hkl, "delta_theta_arcsec"].iloc[0])
    source_rows = []
    partner_rows = []
    assignment_rows = []
    for index in range(n):
        source_id = 1000 + index
        partner_id = 2000 if reused_partner else 2000 + index
        theta_source = 120000.0 + index * 0.25
        theta_partner = theta_source + theory + (index - (n - 1) / 2.0) * 0.02
        source_rows.append({
            "peak_id": source_id, "branch_id": 0, "x": index, "y": 0,
            "theta": theta_source, "theta_index": index,
        })
        if not reused_partner or index == 0:
            partner_rows.append({
                "peak_id": partner_id, "branch_id": 1, "x": index, "y": 1,
                "theta": theta_partner if not reused_partner else 120000.0 + theory,
                "theta_index": index + 1,
            })
        assignment_rows.append({
            "source_peak_id": source_id, "source_branch_id": 0,
            "assignment_class": "partner", "assigned_partner_branch_id": 1,
            "matched_partner_peak_id": partner_id,
        })
    catalog = pd.DataFrame(source_rows + partner_rows)
    assignments = pd.DataFrame(assignment_rows)
    correlation_tables = {
        "branch_correlation_pairs": pd.DataFrame([{
            "source_branch_id": 0, "partner_branch_id": 1,
            "theta_shift_final_arcsec": -theory,
        }]),
        "branch_correlation_assignments": assignments,
    }
    return catalog, correlation_tables, theory


def test_hkl_operation_ranks_known_caf2_reflection_from_exact_peak_id_pairs():
    catalog, correlation_tables, expected_delta = _catalog_for_reflection("(004)")
    result = default_registry().execute(
        "theory.identify_hkl",
        {
            "branches": ArtifactBundle(
                tables={"branch_catalog": catalog}, metadata={"theta_unit": "arcsec"}
            ),
            "branch_correlation": ArtifactBundle(tables=correlation_tables),
        },
        {},
    )
    stats = result.bundle.tables["hkl_pair_statistics"].iloc[0]
    ranking = result.bundle.tables["hkl_pair_ranking"]
    assert stats["pair_id"] == "0↔1"
    assert stats["matched_assignment_count"] == 7
    assert stats["best_hkl"] == "(004)"
    assert stats["best_theory_delta_theta_arcsec"] == pytest.approx(expected_delta)
    assert stats["best_abs_difference_arcsec"] < 0.1
    assert result.bundle.metadata["theory"]["ranking_is_probability"] is False
    assert result.bundle.metadata["theory"]["radiation"] == "parameterized Kα1/Kα2"
    assert result.bundle.tables["hkl_pair_measurements"]["pair_id"].nunique() == 1
    assert len(ranking) == len(result.bundle.tables["hkl_theoretical_reflections"])


def test_many_to_one_partner_assignments_are_preserved_and_reported():
    catalog, correlation_tables, _ = _catalog_for_reflection("(004)", n=5, reused_partner=True)
    measurements, stats = identify_kalpha_hkl(
        catalog,
        correlation_tables,
        branch_metadata={"theta_unit": "arcsec"},
        config=KAlphaIdentificationConfig(),
    )[2:4]
    assert len(measurements) == 5
    assert measurements["matched_partner_peak_id"].nunique() == 1
    assert int(stats.iloc[0]["reused_partner_assignment_count"]) == 4


def test_hkl_operation_artifact_roundtrip_has_stable_empty_and_nonempty_schemas(tmp_path):
    catalog, correlation_tables, _ = _catalog_for_reflection("(004)")
    result = default_registry().execute(
        "theory.identify_hkl",
        {
            "branches": ArtifactBundle(
                tables={"branch_catalog": catalog}, metadata={"theta_unit": "arcsec"}
            ),
            "branch_correlation": ArtifactBundle(tables=correlation_tables),
        },
        {},
    )
    store = ArtifactStore(tmp_path / "artifacts")
    artifact_id = store.write(result.bundle)
    loaded = store.read(artifact_id)
    assert set(loaded.tables) == {
        "hkl_theoretical_reflections", "hkl_bragg_rejected", "hkl_pair_measurements",
        "hkl_pair_statistics", "hkl_pair_ranking",
    }
    assert loaded.tables["hkl_pair_measurements"].equals(
        result.bundle.tables["hkl_pair_measurements"]
    )
    assert loaded.tables["hkl_pair_statistics"].iloc[0]["best_hkl"] == "(004)"


def test_hkl_refuses_unknown_theta_units_and_missing_peak_id_joins():
    catalog, correlation_tables, _ = _catalog_for_reflection("(004)")
    with pytest.raises(ValueError, match="theta_unit='arcsec'"):
        identify_kalpha_hkl(
            catalog, correlation_tables, branch_metadata={"theta_unit": "degrees"},
            config=KAlphaIdentificationConfig(),
        )
    broken = {key: frame.copy() for key, frame in correlation_tables.items()}
    broken["branch_correlation_assignments"].loc[0, "matched_partner_peak_id"] = 999999
    with pytest.raises(ValueError, match="нет в branch_catalog"):
        identify_kalpha_hkl(
            catalog, broken, branch_metadata={"theta_unit": "arcsec"},
            config=KAlphaIdentificationConfig(),
        )


def test_hkl_validation_rejects_nonphysical_parameters():
    with pytest.raises(ValueError, match=r"λ\(Kα2\) > λ\(Kα1\)"):
        _validate_kalpha_hkl({"lambda_ka1_A": 0.72, "lambda_ka2_A": 0.71})
    with pytest.raises(ValueError, match="max_miller_index"):
        _validate_kalpha_hkl({"max_miller_index": 0})


def test_bragg_candidate_enumeration_obeys_f_centering_and_bragg_limit():
    reflections, rejected = calculate_doublet_reflections(
        CA_F2_FLUORITE, BraggDoubletConfig(max_miller_index=10)
    )
    assert not reflections.empty
    for row in reflections.itertuples(index=False):
        assert row.h % 2 == row.k % 2 == row.l % 2
        assert row.d_A >= 0.7135900 / 2.0
        assert row.theta_ka2_deg > row.theta_ka1_deg
        assert row.delta_theta_arcsec > 0
    assert set(rejected.columns) == {"h", "k", "l", "hkl", "d_A", "reason"}


def test_hkl_labels_are_unambiguous_for_multi_digit_indices():
    from roc_mcs.theory.bragg import _format_hkl

    assert _format_hkl((1, 1, 1)) == "(111)"
    assert _format_hkl((10, 10, 10)) == "(10 10 10)"
    assert _format_hkl((2, 10, 12)) == "(2 10 12)"


def test_crystal_model_registry_rejects_duplicate_identifier():
    from roc_mcs.theory.crystals import register_crystal_model

    with pytest.raises(ValueError, match="уже зарегистрирована"):
        register_crystal_model(CA_F2_FLUORITE)


def test_caf2_crystalpy_adapter_preserves_notebook_amplitudes_and_widths(monkeypatch, tmp_path):
    import sys
    import types

    from roc_mcs.theory import crystalpy_caf2

    class FakeBraggDiffraction:
        pass

    setup_kwargs = []

    class FakeSetup:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            setup_kwargs.append(kwargs)

        def F0(self, _energy):
            return 2.0 + 1.0j

        def FH(self, _energy):
            return 3.0 + 4.0j

        def FH_bar(self, _energy):
            return 1.0 - 2.0j

        def psiH(self, _energy):
            return 0.1 + 0.2j

        def psiH_bar(self, _energy):
            return 0.3 - 0.4j

        def angleBragg(self, energy):
            return 0.1 if energy > 17000 else 0.101

        def darwinHalfwidthS(self, _energy):
            return 1.0e-6

        def darwinHalfwidthP(self, _energy):
            return 2.0e-6

    crystalpy_package = types.ModuleType("crystalpy")
    crystalpy_package.__path__ = []
    diffraction_package = types.ModuleType("crystalpy.diffraction")
    diffraction_package.__path__ = []
    geometry_module = types.ModuleType("crystalpy.diffraction.GeometryType")
    geometry_module.BraggDiffraction = FakeBraggDiffraction
    setup_module = types.ModuleType("crystalpy.diffraction.DiffractionSetupXraylib")
    setup_module.DiffractionSetupXraylib = FakeSetup
    monkeypatch.setitem(sys.modules, "crystalpy", crystalpy_package)
    monkeypatch.setitem(sys.modules, "crystalpy.diffraction", diffraction_package)
    monkeypatch.setitem(sys.modules, "crystalpy.diffraction.GeometryType", geometry_module)
    monkeypatch.setitem(sys.modules, "crystalpy.diffraction.DiffractionSetupXraylib", setup_module)
    monkeypatch.setattr(crystalpy_caf2, "crystalpy_available", lambda: True)

    base, _ = calculate_doublet_reflections(CA_F2_FLUORITE, BraggDoubletConfig(max_miller_index=2))
    enriched = crystalpy_caf2.add_crystalpy_properties(
        base.head(1), lattice_a_A=5.4635,
        lambda_ka1_A=0.7093000, lambda_ka2_A=0.7135900,
        thickness_m=2.5e-5,
    )
    assert setup_kwargs[0]["thickness"] == pytest.approx(2.5e-5)
    row = enriched.iloc[0]
    assert row["FH_real"] == pytest.approx(3.0)
    assert row["FH_imag"] == pytest.approx(4.0)
    assert row["FH_sq"] == pytest.approx(25.0)
    assert row["psiH_bar_imag"] == pytest.approx(-0.4)
    assert row["darwin_sigma_arcsec"] > 0
    assert row["darwin_pi_arcsec"] > row["darwin_sigma_arcsec"]
    assert row["crystalpy_thickness_m"] == pytest.approx(2.5e-5)

    store = ArtifactStore(tmp_path / "crystalpy_artifact")
    loaded = store.read(store.write(ArtifactBundle(tables={"reflections": enriched})))
    roundtrip = loaded.tables["reflections"].iloc[0]
    assert roundtrip["FH_imag"] == pytest.approx(4.0)
    assert roundtrip["psiH_real"] == pytest.approx(0.1)


def test_caf2_crystalpy_adapter_fails_explicitly_without_optional_dependencies(monkeypatch):
    from roc_mcs.theory import crystalpy_caf2

    monkeypatch.setattr(crystalpy_caf2, "crystalpy_available", lambda: False)
    with pytest.raises(RuntimeError, match="crystalpy и xraylib"):
        crystalpy_caf2.add_crystalpy_properties(
            pd.DataFrame([{"h": 1, "k": 1, "l": 1}]),
            lattice_a_A=5.4635, lambda_ka1_A=0.7093000, lambda_ka2_A=0.7135900,
        )


def test_histogram_bin_width_is_view_only_and_legacy_parameter_is_ignored():
    validated = _validate_kalpha_hkl({"histogram_bin_width_arcsec": -1000.0})
    assert "histogram_bin_width_arcsec" not in validated
    assert validated["lattice_a_A"] == pytest.approx(5.4635)


def test_database_crystal_uses_authoritative_unit_cell_a_not_stale_caf2_value(monkeypatch):
    from types import SimpleNamespace

    from roc_mcs.theory import crystals

    model = SimpleNamespace(
        display_name="Quartz · XrayLib",
        unit_cell_parameters_A={"a": 4.913, "b": 4.913, "c": 5.405},
    )
    monkeypatch.setattr(crystals, "get_crystal_model", lambda _model_id: model)
    validated = _validate_kalpha_hkl({
        "crystal_model_id": "xraylib:quartz",
        "lattice_a_A": 5.4635,  # stale value from the old disabled CaF2 control
    })
    assert validated["lattice_a_A"] == pytest.approx(4.913)
    assert "histogram_bin_width_arcsec" not in validated


def test_bragg_doublet_separation_is_the_difference_of_bragg_theta_angles():
    from roc_mcs.theory.bragg import RAD_TO_ARCSEC

    reflections, _ = calculate_doublet_reflections(
        CA_F2_FLUORITE, BraggDoubletConfig(max_miller_index=10)
    )
    row = reflections.loc[reflections["hkl"] == "(004)"].iloc[0]
    d = CA_F2_FLUORITE.d_spacing_A((0, 0, 4), 5.4635)
    theta_1 = np.arcsin(0.7093000 / (2.0 * d))
    theta_2 = np.arcsin(0.7135900 / (2.0 * d))
    expected_arcsec = (theta_2 - theta_1) * RAD_TO_ARCSEC
    assert row["theta_ka1_deg"] == pytest.approx(np.degrees(theta_1))
    assert row["theta_ka2_deg"] == pytest.approx(np.degrees(theta_2))
    assert row["delta_theta_arcsec"] == pytest.approx(expected_arcsec)


def test_general_triclinic_xraylib_cell_metric_matches_reciprocal_metric():
    from roc_mcs.theory.crystals import XraylibCrystalModel

    cell = {
        "a": 4.2, "b": 5.1, "c": 6.3,
        "alpha": 76.0, "beta": 83.0, "gamma": 68.0,
    }
    model = XraylibCrystalModel(
        model_id="test:triclinic", crystal_name="triclinic",
        crystal_data=cell, display_name="test triclinic",
    )
    alpha, beta, gamma = np.radians([cell["alpha"], cell["beta"], cell["gamma"]])
    a, b, c = cell["a"], cell["b"], cell["c"]
    direct_metric = np.array([
        [a * a, a * b * np.cos(gamma), a * c * np.cos(beta)],
        [a * b * np.cos(gamma), b * b, b * c * np.cos(alpha)],
        [a * c * np.cos(beta), b * c * np.cos(alpha), c * c],
    ])
    reciprocal_metric = np.linalg.inv(direct_metric)
    for hkl in ((1, 0, 0), (0, 1, 1), (2, -1, 3)):
        indices = np.asarray(hkl, dtype=float)
        expected = 1.0 / np.sqrt(indices @ reciprocal_metric @ indices)
        assert model.d_spacing_A(hkl) == pytest.approx(expected, rel=1e-12)


def test_artifact_store_reads_only_requested_tables(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    bundle = ArtifactBundle(
        arrays={"large_array": np.arange(1000, dtype=np.float64)},
        tables={
            "requested": pd.DataFrame({"value": [1, 2, 3]}),
            "not_requested": pd.DataFrame({"ignored": [4, 5]}),
        },
        metadata={"provenance": "test"},
    )
    artifact_id = store.write(bundle)

    selected = store.read_tables(artifact_id, ("requested",))

    assert selected.arrays == {}
    assert set(selected.tables) == {"requested"}
    assert selected.tables["requested"].equals(bundle.tables["requested"])
    assert selected.metadata == {"provenance": "test"}
    with pytest.raises(KeyError, match="not_in_artifact"):
        store.read_tables(artifact_id, ("not_in_artifact",))
    with pytest.raises(ValueError, match="хотя бы одну"):
        store.read_tables(artifact_id, ())


def test_artifact_store_limits_preview_rows_per_group_without_changing_artifact(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    ranking = pd.DataFrame({
        "pair_id": ["A", "A", "A", "B", "B", "B"],
        "rank": [1, 2, 3, 1, 2, 3],
        "hkl": ["(100)", "(110)", "(111)", "(200)", "(210)", "(211)"],
    })
    artifact_id = store.write(ArtifactBundle(tables={"ranking": ranking}))

    preview = store.read_tables(
        artifact_id, ("ranking",), group_head_limits={"ranking": ("pair_id", 2)}
    )
    full = store.read(artifact_id)

    assert preview.tables["ranking"].to_dict(orient="records") == ranking.iloc[
        [0, 1, 3, 4]
    ].to_dict(orient="records")
    assert len(full.tables["ranking"]) == 6
    assert full.tables["ranking"].equals(ranking)


def test_artifact_store_rejects_boolean_group_head_limit(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact_id = store.write(ArtifactBundle(tables={
        "ranking": pd.DataFrame({"pair_id": ["A"], "rank": [1]}),
    }))

    with pytest.raises(ValueError, match="целый лимит"):
        store.read_tables(
            artifact_id, ("ranking",), group_head_limits={"ranking": ("pair_id", True)}
        )


def test_artifact_store_group_head_limit_handles_empty_table(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact_id = store.write(ArtifactBundle(tables={
        "ranking": pd.DataFrame({"pair_id": pd.Series(dtype=object), "rank": pd.Series(dtype=int)}),
    }))

    preview = store.read_tables(
        artifact_id, ("ranking",), group_head_limits={"ranking": ("pair_id", 15)}
    )

    assert preview.tables["ranking"].empty
    assert list(preview.tables["ranking"].columns) == ["pair_id", "rank"]
