"""Baixa todas as anotações disponíveis em train/valid/test, sem os frames de imagem."""
import gzip
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests
from range_zip import HTTPRangeFile

ROOT = Path(__file__).resolve().parent
REPO = "SoccerNet/SN-GSR-2024"


def run():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "results/download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        revision = manifest["revision"]
    else:
        r = requests.get(f"https://huggingface.co/api/datasets/{REPO}", timeout=90)
        r.raise_for_status()
        info = r.json()
        (raw / "huggingface_repo.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        revision = info["sha"]
        manifest = {"repo": REPO, "revision": revision, "checked_date": "2026-11-12",
                    "scope": "All Labels-GameState.json in train, valid and test; no image frames or challenge split",
                    "splits": {}, "files": []}
    r = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}", timeout=90)
    r.raise_for_status()
    tree = r.json()
    (raw / "huggingface_tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    for split in ["train", "valid", "test"]:
        entry = next(e for e in tree if e["path"] == f"{split}.zip")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{split}.zip?download=true"
        folder = raw / split
        folder.mkdir(exist_ok=True)
        remote = HTTPRangeFile(url, entry["size"])
        with zipfile.ZipFile(remote) as z:
            labels = sorted((i for i in z.infolist() if i.filename.endswith("Labels-GameState.json")), key=lambda x:x.filename)
            manifest["splits"][split] = {"archive_bytes": entry["size"], "archive_lfs_sha256": entry.get("lfs", {}).get("oid"),
                                         "annotation_files": len(labels), "annotation_compressed_bytes": sum(i.compress_size for i in labels)}
            for ix, item in enumerate(labels, 1):
                clip = Path(item.filename).parts[-2]
                target = folder / f"{clip}.json.gz"
                previous = next((m for m in manifest["files"] if m["split"] == split and m["clip"] == clip), None)
                if previous and target.exists():
                    continue
                last_error = None
                for attempt in range(3):
                    try:
                        data = z.read(item)  # ZipFile confere o CRC do membro.
                        break
                    except (requests.RequestException, IOError) as e:
                        last_error = e
                        if attempt == 2:
                            raise
                        time.sleep(2)
                obj = json.loads(data)
                if float(obj["info"]["version"]) < 1.3:
                    raise ValueError("Anotações anteriores à revisão de consistência temporal v1.3.")
                compressed = gzip.compress(data, compresslevel=1, mtime=0)
                target.write_bytes(compressed)
                record = {"split": split, "clip": clip, "member": item.filename,
                          "bytes_uncompressed": len(data), "sha256_json": hashlib.sha256(data).hexdigest(),
                          "zip_crc32": item.CRC, "annotation_version": obj["info"]["version"],
                          "frames": len(obj["images"]), "annotations": len(obj["annotations"])}
                manifest["files"] = [m for m in manifest["files"] if not (m["split"] == split and m["clip"] == clip)] + [record]
                manifest["download_complete"] = False
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"{split}: {ix}/{len(labels)} {clip}, v{obj['info']['version']}", flush=True)
            manifest["splits"][split]["bytes_transferred_this_run"] = remote.transferred
        remote.close()
    manifest["download_complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Download completo: {len(manifest['files'])} arquivos de anotações.", flush=True)
    return manifest


if __name__ == "__main__":
    run()
