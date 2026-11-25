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
