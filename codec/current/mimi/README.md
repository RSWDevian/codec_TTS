We use HuggingFace `transformers.MimiModel` (`kyutai/mimi`) directly rather than vendoring Kyutai's own `moshi` package — see `codec/current/wrapper.py`.

Reserved for vendored/patched Mimi code if we ever need to modify the codec itself (e.g. custom streaming behavior) beyond what the `transformers` integration exposes.
