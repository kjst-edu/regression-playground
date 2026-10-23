"""Pack a data directory and the app into a static site.

    uv run pack.py DATA_DIR OUT_DIR

DATA_DIR holds app.toml and the data file named in its [data] file. The spec is
checked first; on a mistake this prints the reason, exits 1 and writes nothing.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "app"))

from playground.spec import SpecError, check, load_data, load_spec  # noqa: E402

# stargazer is pure Python. It is vendored into the exported app (its package
# directory sits next to app.py) so the page never asks PyPI for it at load time.
STARGAZER_URL = (
    "https://files.pythonhosted.org/packages/af/49/0f0bc34f94e6ba65f8e5cc06208d11759153f02fb61f4a9f9db98e86bb68/"
    "stargazer-0.0.7-py3-none-any.whl"
)
STARGAZER_SHA256 = "0bd47e7fd0a41317724743a63f6c92472d27adf8ab12d7e4d2682fc31aa51bca"
CACHE = ROOT / "_build"


def stargazer_wheel() -> bytes:
    cached = CACHE / "stargazer-0.0.7-py3-none-any.whl"
    if cached.exists():
        data = cached.read_bytes()
    else:
        data = urllib.request.urlopen(STARGAZER_URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != STARGAZER_SHA256:
        raise SystemExit("stargazer wheel: sha256 mismatch")
    CACHE.mkdir(exist_ok=True)
    cached.write_bytes(data)
    return data


def vendor_stargazer(app_dir: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(stargazer_wheel())) as z:
        for name in z.namelist():
            if name.startswith("stargazer/") and not name.endswith("/"):
                target = app_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
        (app_dir / "stargazer" / "LICENSE").write_bytes(
            z.read("stargazer-0.0.7.dist-info/LICENSE")
        )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: uv run pack.py DATA_DIR OUT_DIR", file=sys.stderr)
        return 2
    data_dir, out_dir = Path(argv[0]), Path(argv[1])
    toml_path = data_dir / "app.toml"
    try:
        spec = load_spec(toml_path)
        df = load_data(spec, data_dir)
        check(spec, df)
    except (SpecError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    from shinylive._export import export

    with tempfile.TemporaryDirectory() as tmp:
        app_dir = Path(tmp) / "app"
        shutil.copytree(
            ROOT / "app",
            app_dir,
            ignore=shutil.ignore_patterns("__pycache__", "app.toml", "data.*"),
        )
        shutil.copy(toml_path, app_dir / "app.toml")
        data_target = app_dir / spec.file
        data_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(data_dir / spec.file, data_target)
        vendor_stargazer(app_dir)
        reqs = (app_dir / "requirements.txt").read_text().splitlines()
        (app_dir / "requirements.txt").write_text(
            "\n".join(r for r in reqs if not r.startswith("stargazer")) + "\n"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        export(app_dir, out_dir, template_params={"title": spec.title or "Regression playground"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
