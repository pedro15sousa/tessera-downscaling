"""Tests for TESSERA embedding extraction.

These tests verify the extraction logic without requiring ``geotessera``
to be installed or network access. The ``geotessera`` dependency is mocked
where needed. The caching utilities are tested with real file I/O.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tessera_downscaling.data.tessera import (
    load_cached_embeddings,
    save_cached_embeddings,
)


def test_save_and_load_cached_embeddings(tmp_path):
    """Round-trip test: save embeddings, then load them back."""
    embeddings = np.random.randn(10, 128).astype(np.float32)
    cache_path = tmp_path / "test_embeddings.npy"

    save_cached_embeddings(embeddings, cache_path)
    assert cache_path.exists()

    loaded = load_cached_embeddings(cache_path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, embeddings)


def test_load_cached_embeddings_missing_file(tmp_path):
    """Loading from a non-existent path returns None."""
    result = load_cached_embeddings(tmp_path / "nonexistent.npy")
    assert result is None


def test_save_creates_parent_directories(tmp_path):
    """Saving to a nested path creates intermediate directories."""
    embeddings = np.random.randn(5, 128).astype(np.float32)
    cache_path = tmp_path / "nested" / "dir" / "embeddings.npy"

    save_cached_embeddings(embeddings, cache_path)
    assert cache_path.exists()

    loaded = np.load(cache_path)
    np.testing.assert_array_equal(loaded, embeddings)


def test_extract_point_embeddings_without_geotessera():
    """When geotessera is not installed, extraction returns zeros."""
    # Temporarily make geotessera unimportable.
    with patch.dict("sys.modules", {"geotessera": None}):
        from tessera_downscaling.data.tessera import extract_point_embeddings

        lats = np.array([51.5, 48.8, 52.5])
        lons = np.array([-0.1, 2.3, 13.4])

        result = extract_point_embeddings(lats, lons, year=2024)

        assert result.shape == (3, 128)
        assert np.all(result == 0)


def test_extract_patch_embeddings_without_geotessera():
    """When geotessera is not installed, patch extraction returns zeros."""
    with patch.dict("sys.modules", {"geotessera": None}):
        from tessera_downscaling.data.tessera import extract_patch_embeddings

        lats = np.array([51.5, 48.8])
        lons = np.array([-0.1, 2.3])

        result = extract_patch_embeddings(lats, lons, year=2024, patch_size=32)

        assert result.shape == (2, 32, 32, 128)
        assert np.all(result == 0)

def test_load_stations_ghcnh_format(tmp_path):
    """Station loading works with GHCNh station list column names."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from extract_tessera import load_stations

    csv_path = tmp_path / "stations.csv"
    csv_path.write_text(
        "GHCN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME\n"
        "UK000000001,51.5,-0.1,10.0,,London\n"
        "FR000000001,48.8,2.3,35.0,,Paris\n"
        "US000000001,40.7,-74.0,10.0,,New York\n"
        "JP000000001,35.7,139.7,40.0,,Tokyo\n"
    )

    # Without filter: all 4 stations.
    df = load_stations(csv_path)
    assert len(df) == 4
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert "station_id" in df.columns

    # With European filter: only London and Paris.
    df_eu = load_stations(csv_path, region="europe")
    assert len(df_eu) == 2
    assert set(df_eu["station_id"]) == {"UK000000001", "FR000000001"}