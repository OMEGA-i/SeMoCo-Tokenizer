"""Generate the SOMA77 FK template consumed by the ``fk_geom`` training loss.

Writes ``{parents[77], offsets[77,3]}`` (a small constant ``.npz``) that
:mod:`losses.fk_geom` loads at training time. Requires the ``[soma]`` extras
and the SOMA-X submodule; the SOMA mean body ships with the submodule, so no
licensed SMPL-X download is needed.

    python -m tools.build_soma77_template [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from losses.fk_geom import TEMPLATE_PATH, build_soma77_template


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", default=str(TEMPLATE_PATH),
                   help="output .npz path (default: %(default)s)")
    p.add_argument("--device", default="cpu", help="FK device (cpu / cuda)")
    args = p.parse_args(argv)

    out = build_soma77_template(Path(args.out), device=args.device)
    print(f"[fk-template] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
