import json
from unittest import mock

import pytest

from diffusers.commands.fp16_safetensors import FP16SafetensorsCommand
from diffusers.loaders.single_file_utils import fetch_original_config


def _open_with_cp936_default(real_open):
    # Emulate a Windows machine whose locale encoding is not UTF-8 (e.g. cp936).
    def _open(file, mode="r", *args, **kwargs):
        if "b" not in mode and kwargs.get("encoding") is None:
            kwargs["encoding"] = "gbk"
        return real_open(file, mode, *args, **kwargs)

    return _open


def test_fetch_original_config_reads_utf8(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("# 中文注释 — é\nmodel:\n  target: foo\n", encoding="utf-8")

    with mock.patch("builtins.open", _open_with_cp936_default(open)):
        config = fetch_original_config(str(config_path))

    assert config["model"]["target"] == "foo"


def test_fp16_safetensors_reads_model_index_utf8(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Foo", "description": "é中文—"}, ensure_ascii=False),
        encoding="utf-8",
    )
    command = FP16SafetensorsCommand(ckpt_id="dummy", fp16=True, use_safetensors=True)
    model_index = str(tmp_path / "model_index.json")

    with (
        mock.patch("builtins.open", _open_with_cp936_default(open)),
        mock.patch("diffusers.commands.fp16_safetensors.hf_hub_download", return_value=model_index),
        mock.patch("diffusers.commands.fp16_safetensors.import_module") as import_module,
    ):
        import_module.return_value.Foo.from_pretrained.side_effect = RuntimeError("stop")
        with pytest.raises(RuntimeError, match="stop"):
            command.run()
