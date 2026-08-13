"""Sale Deed AI inference server.

Built around the locally trained Gemma 3 4B deeds model in `AI server/gemma4b`.
No model is ever downloaded, substituted or retrained by this package.

Note on naming: the directory `AI server/` (with a space) holds model *weights*.
This package, `ai_server/`, holds the serving *code*. They are distinct.
"""

__all__ = ["hardware", "profiles"]
__version__ = "3.0.0"
