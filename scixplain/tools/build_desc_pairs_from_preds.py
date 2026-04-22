#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def norm(s: str | None) -> str:
    if s is None:
        return ''
    return ' '.join(str(s).replace('\u00a0', ' ').split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_jsonl', required=True)
    ap.add_argument('--gt_desc_jsonl', required=True)
    ap.add_argument('--out_jsonl', required=True)
    ap.add_argument('--scale', default='desc')
    args = ap.parse_args()

    gt_uid = {}
    gt_ip = {}
    for r in load_jsonl(Path(args.gt_desc_jsonl)):
        uid = r.get('uid')
        ip = r.get('image_path')
        desc = norm(r.get('description'))
        if not desc:
            continue
        if uid:
            gt_uid[uid] = desc
        if ip:
            gt_ip[ip] = desc

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    with out_path.open('w', encoding='utf-8') as wf:
        for r in load_jsonl(Path(args.pred_jsonl)):
            n_in += 1
            if 'error' in r:
                continue
            p = norm(r.get('pred_description'))
            if not p:
                continue
            uid = r.get('uid')
            ip = r.get('image_path')
            ref = gt_uid.get(uid, '') if uid else ''
            if (not ref) and ip:
                ref = gt_ip.get(ip, '')
            if not ref:
                continue
            row = {
                'uid': uid,
                'image_path': ip,
                'scale': args.scale,
                'pred': p,
                'ref': ref,
            }
            wf.write(json.dumps(row, ensure_ascii=False) + '\n')
            n_out += 1

    print(json.dumps({'in_rows': n_in, 'out_rows': n_out, 'out_jsonl': str(out_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
