#!/usr/bin/env python3
"""Download SoccerNet-MVFoul without hard-coding the NDA password."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SoccerNet-MVFoul data.")
    parser.add_argument("--output", required=True, help="Directory where SoccerNet data will be stored.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test", "challenge"], help="Splits to download.")
    parser.add_argument("--version", default=None, help='Optional MVFoul video version, for example "720p".')
    parser.add_argument("--password-env", default="SOCCERNET_PASSWORD", help="Environment variable containing the password.")
    args = parser.parse_args()

    password = os.environ.get(args.password_env)
    if not password:
        password = getpass.getpass("SoccerNet password: ")

    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=args.output)
    kwargs = {"task": "mvfouls", "split": args.splits, "password": password}
    if args.version:
        kwargs["version"] = args.version
    downloader.downloadDataTask(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
