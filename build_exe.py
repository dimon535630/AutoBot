from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PyInstaller.__main__ import run
from PyInstaller.utils.hooks import collect_all


ROOT = Path(__file__).resolve().parent
EXCLUDE_DIRS = {".git", ".github", "build", "dist", "__pycache__", ".venv", "venv"}
EXCLUDE_SUFFIXES = {".py", ".pyc", ".pyo", ".md", ".png", ".jpg", ".jpeg", ".mp3", ".json", ".db"}


def iter_data_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            continue

        yield path


def make_add_data_args(root: Path) -> list[str]:
    args: list[str] = []
    for file_path in iter_data_files(root):
        rel_dir = file_path.relative_to(root).parent
        target_dir = "." if str(rel_dir) == "." else str(rel_dir).replace("\\", "/")
        args.extend(["--add-data", f"{file_path};{target_dir}"])
    return args


def main() -> None:
    modules_with_plugins = [
        "cv2",
        "pygame",
        "pyautogui",
        "pydirectinput",
        "PIL",
        "keyboard",
        "numpy",
    ]

    all_args = [
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onefile",
        "--name",
        "FishingBot",
    ]

    all_args.extend(make_add_data_args(ROOT))

    for module_name in modules_with_plugins:
        datas, binaries, hiddenimports = collect_all(module_name)
        for src, dst in datas:
            all_args.extend(["--add-data", f"{src};{dst}"])
        for src, dst in binaries:
            all_args.extend(["--add-binary", f"{src};{dst}"])
        for hidden in hiddenimports:
            all_args.extend(["--hidden-import", hidden])

    all_args.append("launcher.py")

    print("PyInstaller arguments prepared (code/modules in EXE, assets external):")
    print(f"  data files from project: {len([a for a in all_args if a == '--add-data'])}")
    print(f"  binaries collected: {len([a for a in all_args if a == '--add-binary'])}")
    print(f"  hidden imports: {len([a for a in all_args if a == '--hidden-import'])}")

    run(all_args)


if __name__ == "__main__":
    main()
