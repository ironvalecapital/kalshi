from __future__ import annotations

import inspect

from kalshi_bot import cli


def test_watch_flow_uses_per_side_depth_not_split_total() -> None:
    src = inspect.getsource(cli.watch_flow)
    assert "depth_yes_topk(3)" in src
    assert "depth_no_topk(3)" in src
    assert "depth_topk(3)//2" not in src

