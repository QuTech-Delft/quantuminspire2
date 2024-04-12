from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import quantuminspire.util.configuration as configuration


@pytest.fixture
def mocked_config_file(mocker: MockerFixture) -> Any:
    config_file = mocker.patch("quantuminspire.util.configuration.Path")()
    config_file.exists.return_value = True
    config_file.read_text.return_value = '{"auths": {"https://host": {"well_known_endpoint": "https://some_url"}}}'
    return config_file


def test_force_file_into_existence_file_does_not_exist(mocked_config_file: MagicMock) -> None:
    mocked_config_file.exists.return_value = False
    open_mock = MagicMock()
    mocked_config_file.open.return_value = open_mock
    configuration.ensure_config_file_exists(mocked_config_file, "utf-8")
    mocked_config_file.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mocked_config_file.open.assert_called_once_with("w", encoding="utf-8")
    open_mock.write.assert_called_once_with(
        """
{
  "auths": {
    "https://staging.qi2.quantum-inspire.com": {
      "well_known_endpoint":  "https://auth.qi2.quantum-inspire.com/realms/oidc_staging/.well-known/openid-configuration"
    },
    "https://api.qi2.quantum-inspire.com": {
      "well_known_endpoint":  "https://auth.qi2.quantum-inspire.com/realms/oidc_production/.well-known/openid-configuration"
    }
  }
}
"""
    )


def test_force_file_into_existence_file_exists() -> None:
    path = MagicMock()
    path.exists.return_value = True
    open_mock = MagicMock()
    path.open.return_value = open_mock
    configuration.ensure_config_file_exists(path, "utf-8")
    path.parent.mkdir.assert_not_called()
    path.open.assert_not_called()
    path.open().write.assert_not_called()
    open_mock.write.assert_not_called()


def test_json_config_settings_file_does_not_exist(mocked_config_file: MagicMock) -> None:
    mocked_config_file.exists.return_value = False

    assert configuration.JsonConfigSettingsSource(configuration.Settings)() == {
        "auths": {
            "https://host": {
                "well_known_endpoint": "https://some_url",
            },
        },
    }


def test_json_config_settings_file_does_exist(mocked_config_file: MagicMock) -> None:
    assert configuration.JsonConfigSettingsSource(configuration.Settings)() == {
        "auths": {
            "https://host": {
                "well_known_endpoint": "https://some_url",
            },
        },
    }


def test_json_config_settings_qi2_813(mocked_config_file: MagicMock) -> None:

    settings = configuration.Settings()

    assert settings.auths["https://host"].well_known_endpoint == "https://some_url"


def test_customise_sources(mocker: MockerFixture) -> None:
    init_settings = MagicMock()
    env_settings = MagicMock()
    dot_env_settings = MagicMock()
    file_secret_settings = MagicMock()
    json_settings_source = mocker.patch("quantuminspire.util.configuration.JsonConfigSettingsSource")
    assert configuration.Settings.settings_customise_sources(
        configuration.Settings, init_settings, env_settings, dot_env_settings, file_secret_settings
    ) == (
        init_settings,
        env_settings,
        json_settings_source(configuration.Settings),
        file_secret_settings,
    )


def test_settings_from_init() -> None:
    settings = configuration.Settings(auths={"https://example.com": {"well_known_endpoint": "https://some_url/"}})
    assert (
        settings.auths.items()
        >= {"https://example.com": configuration.AuthSettings(well_known_endpoint="https://some_url/")}.items()
    )


def test_tokeninfo() -> None:
    tokeninfo = configuration.TokenInfo(
        access_token="secret",
        expires_in=100,
        refresh_token="also_secret",
        refresh_expires_in=200,
        generated_at=10000,
    )
    assert tokeninfo.expires_at == 10100
    assert tokeninfo.refresh_expires_at == 10200


def test_store_tokens(mocked_config_file: MagicMock) -> None:
    settings = configuration.Settings()
    tokeninfo = configuration.TokenInfo(
        access_token="secret",
        expires_in=100,
        refresh_token="also_secret",
        refresh_expires_in=200,
        generated_at=10000,
    )
    settings.store_tokens("https://host", tokeninfo)
