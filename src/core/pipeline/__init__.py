"""Document processing pipeline.

    PDF -> images -> OCR -> cleanup -> extract -> validate -> translate -> CSV

Split deliberately in two:

`stages` holds the document-processing logic and depends on **no database**. Each
stage takes text or a path and returns a result, so the whole chain is testable on
a machine with no PostgreSQL server.

`runner` holds the database-driven orchestration: claiming work, per-stage resume,
retry caps, the batch queue, and start/pause/stop.

That split is not cosmetic. It means the part that decides *what the data is* can
be verified independently of the part that decides *when work happens*.
"""

from .stages import (
    ExtractStage,
    OcrStage,
    StageName,
    StageOutcome,
    TranslateStage,
    ValidateStage,
)

__all__ = [
    "ExtractStage",
    "OcrStage",
    "StageName",
    "StageOutcome",
    "TranslateStage",
    "ValidateStage",
]
