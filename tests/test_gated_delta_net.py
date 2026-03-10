import pytest

from exllamav3.modules import gated_delta_net as gdn


def test_should_use_fla_recurrent_auto_prefers_decode_only(monkeypatch):
    monkeypatch.setattr(gdn, "_gdn_recurrent_backend", "auto")
    monkeypatch.setattr(gdn, "fused_recurrent_gated_delta_rule_fwd", object())

    assert gdn.should_use_fla_recurrent(1) is True
    assert gdn.should_use_fla_recurrent(2) is False


def test_should_use_fla_recurrent_ext_disables_fla(monkeypatch):
    monkeypatch.setattr(gdn, "_gdn_recurrent_backend", "ext")
    monkeypatch.setattr(gdn, "fused_recurrent_gated_delta_rule_fwd", object())

    assert gdn.should_use_fla_recurrent(1) is False


def test_should_use_fla_recurrent_fla_requires_dependency(monkeypatch):
    monkeypatch.setattr(gdn, "_gdn_recurrent_backend", "fla")
    monkeypatch.setattr(gdn, "fused_recurrent_gated_delta_rule_fwd", None)

    with pytest.raises(ModuleNotFoundError):
        gdn.should_use_fla_recurrent(1)
