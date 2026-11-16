"""Raw data sources, pinned to exact revisions and checksummed.

Only the official *evaluation* splits are used (dev/validation, plus Qasper's
test set). We never train a model on them; we only tune a handful of
thresholds on our dev slice and report on our held-out test slice.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HF = "https://huggingface.co/datasets"


@dataclass(frozen=True)
class RawFile:
    name: str  # local filename under raw_dir
    url: str
    sha256: str
    extract: str | None = None  # member to extract if the download is a .tgz


SOURCES: dict[str, list[RawFile]] = {
    # MuSiQue "full" = answerable + minimally-edited unanswerable twins. CC BY 4.0
    "musique": [
        RawFile(
            "musique_full_v1.0_dev.jsonl",
            f"{HF}/bdsaglam/musique/resolve/22873a405dd809893b22ada0b499299fb612d2df/musique_full_v1.0_dev.jsonl",
            "8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a",
        )
    ],
    # HotpotQA distractor setting (2 gold + 8 distractor paragraphs). CC BY-SA 4.0
    "hotpotqa": [
        RawFile(
            "hotpot_dist_val.parquet",
            f"{HF}/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet",
            "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
        )
    ],
    # Qasper v0.3, official AllenAI release (full paper text + evidence). CC BY 4.0
    "qasper": [
        RawFile(
            "qasper/qasper-dev-v0.3.json",
            "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
            "2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93",
            extract="qasper-dev-v0.3.json",
        ),
        RawFile(
            "qasper/qasper-test-v0.3.json",
            "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
            "6e29ad410e6e39aa1936017fb965b30a20eb2e7751997f55b97c9d281aa884e5",
            extract="qasper-test-v0.3.json",
        ),
    ],
    # AmbigQA "full" config (includes annotators' search-result snippets). CC BY-SA 3.0
    "ambigqa": [
        RawFile(
            "ambig_full_val.parquet",
            f"{HF}/sewon/ambig_qa/resolve/e969d0132f4dd28c2939d55be34f1788c00ccfe7/full/validation-00000-of-00001.parquet",
            "fc99e779bbfc3049d0f42aaf90eccc055146dd4096d68950d87568215f2a001d",
        )
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(raw_dir: Path, datasets: list[str] | None = None) -> None:
    """Download any missing raw file and verify its checksum."""
    for ds, files in SOURCES.items():
        if datasets and ds not in datasets:
            continue
        for rf in files:
            dest = raw_dir / rf.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                print(f"[download] {ds}: {rf.url}")
                if rf.extract:
                    tmp = dest.parent / (Path(rf.url).name)
                    urllib.request.urlretrieve(rf.url, tmp)
                    with tarfile.open(tmp, "r:gz") as tar:
                        member = tar.getmember(rf.extract)
                        if not member.isfile():
                            raise RuntimeError(f"{rf.extract} is not a regular file")
                        src = tar.extractfile(member)
                        assert src is not None
                        dest.write_bytes(src.read())
                    tmp.unlink()
                else:
                    urllib.request.urlretrieve(rf.url, dest)
            got = sha256(dest)
            if got != rf.sha256:
                raise RuntimeError(f"checksum mismatch for {dest}: {got} != {rf.sha256}")
            print(f"[ok] {ds}: {rf.name}")
