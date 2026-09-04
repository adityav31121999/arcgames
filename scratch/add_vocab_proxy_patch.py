import json
from pathlib import Path

sample_path = Path("notebooks/sample_run_single_game.ipynb")
sub_path = Path("notebooks/submission_run.ipynb")

# Lines to insert into Cell 1, after the "Gemma4Config registration" block
# They must come AFTER gemma4_cfg_cls is defined and BEFORE AutoConfig.register()
vocab_proxy_patch = [
    "\n",
    "# Proxy vocab_size and text_config attrs from Gemma4Config -> text_config so Gemma2 internals don't crash\n",
    "_TEXT_CFG_PROXIED = [\n",
    "    'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',\n",
    "    'num_key_value_heads', 'intermediate_size', 'head_dim', 'rms_norm_eps',\n",
    "]\n",
    "if gemma4_cfg_cls is not None and not hasattr(gemma4_cfg_cls, '_patched_vocab_proxy'):\n",
    "    def _make_proxy(attr):\n",
    "        def _proxy(self):\n",
    "            text_cfg = getattr(self, 'text_config', None)\n",
    "            if text_cfg is not None and not isinstance(text_cfg, dict):\n",
    "                return getattr(text_cfg, attr, None)\n",
    "            if isinstance(text_cfg, dict):\n",
    "                return text_cfg.get(attr)\n",
    "            return None\n",
    "        _proxy.__name__ = attr\n",
    "        return property(_proxy)\n",
    "    for _attr in _TEXT_CFG_PROXIED:\n",
    "        if not hasattr(gemma4_cfg_cls, _attr):\n",
    "            setattr(gemma4_cfg_cls, _attr, _make_proxy(_attr))\n",
    "    gemma4_cfg_cls._patched_vocab_proxy = True\n",
    "    print('🔧 Patched Gemma4Config with text_config attribute proxies.')\n",
    "\n",
    "# Ensure pad_token_id is always int (not list) on init\n",
    "if gemma4_cfg_cls is not None and not hasattr(gemma4_cfg_cls, '_patched_pad_int'):\n",
    "    _orig_g4_init = gemma4_cfg_cls.__init__\n",
    "    def _g4_init_safe(self, *a, **kw):\n",
    "        _orig_g4_init(self, *a, **kw)\n",
    "        for _tok in ('pad_token_id', 'eos_token_id'):\n",
    "            val = getattr(self, _tok, None)\n",
    "            if isinstance(val, (list, tuple)):\n",
    "                setattr(self, _tok, val[0] if val else 1)\n",
    "        if not hasattr(self, 'pad_token_id') or self.pad_token_id is None:\n",
    "            self.pad_token_id = getattr(self, 'eos_token_id', 1)\n",
    "    gemma4_cfg_cls.__init__ = _g4_init_safe\n",
    "    gemma4_cfg_cls._patched_pad_int = True\n",
    "    print('🔧 Patched Gemma4Config pad_token_id to always be int.')\n",
]

for p in [sample_path, sub_path]:
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # Find Cell 1 which has gemma4_cfg_cls and AutoConfig.register
        if "gemma4_cfg_cls" in src and "AutoConfig.register" in src and "_patched_vocab_proxy" not in src:
            # Find the insertion point: after "_patched_pad_token" block, before AutoConfig.register
            new_source = []
            inserted = False
            for i, line in enumerate(cell["source"]):
                if not inserted and "AutoConfig.register" in line:
                    new_source.extend(vocab_proxy_patch)
                    inserted = True
                new_source.append(line)
            cell["source"] = new_source
            print(f"Inserted vocab_proxy patch into Cell 1 of {p.name}")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("Done.")
