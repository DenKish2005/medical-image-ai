"""Download the MIMIC-CXR-JPG images listed in data/mimic_cxr/download_urls.txt (no wget needed).

Asks for your PhysioNet password once (it is not stored). Safe to stop and re-run: files that are
already complete are skipped. Images land in data/mimic_cxr/images/p10/p10000032/s.../<dicom>.jpg,
which is the layout R2Gen expects.

    python download_mimic.py --user YOUR_PHYSIONET_USERNAME
"""
import argparse
import getpass
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

MARK = '/files/mimic-cxr-jpg/2.0.0/files/'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--user', required=True)
    p.add_argument('--urls', default='data/mimic_cxr/download_urls.txt')
    p.add_argument('--out', default='data/mimic_cxr/images')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    urls = [u.strip() for u in Path(a.urls).read_text().splitlines() if u.strip()]
    out = Path(a.out)
    todo = [(u, out / u.split(MARK, 1)[1]) for u in urls]
    todo = [(u, f) for u, f in todo if not (f.exists() and f.stat().st_size > 0)]
    print(f'{len(urls)} images listed, {len(urls) - len(todo)} already downloaded, {len(todo)} to go')
    if not todo:
        return
    auth = (a.user, getpass.getpass(f'PhysioNet password for {a.user}: '))

    def fetch(job):
        url, dest = job
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + '.part')
        for attempt in range(4):
            try:
                with requests.get(url, auth=auth, stream=True, timeout=60, headers={'User-Agent': 'Wget/1.21.4'}) as r:
                    if r.status_code in (401, 403):
                        return url, f'access denied (HTTP {r.status_code})', 0
                    r.raise_for_status()
                    n = 0
                    with open(part, 'wb') as fh:
                        for chunk in r.iter_content(1 << 16):
                            fh.write(chunk)
                            n += len(chunk)
                part.replace(dest)
                return url, None, n
            except Exception as e:  # network hiccup: wait and retry
                err = repr(e)
                time.sleep(3 * (attempt + 1))
        return url, err, 0

    # one request first, so a wrong password or missing access fails immediately and clearly
    url, err, n = fetch(todo[0])
    if err:
        sys.exit(f'First download failed: {err}\nCheck the username/password and that your PhysioNet '
                 f'account has signed the MIMIC-CXR-JPG data use agreement.')
    done, total, failed, t0 = 1, n, [], time.time()
    ex = ThreadPoolExecutor(a.workers)
    futures = [ex.submit(fetch, j) for j in todo[1:]]
    try:
        for fut in as_completed(futures):
                url, err, n = fut.result()
                if err:
                    failed.append((url, err))
                    if 'access denied' in err and len(failed) > 20:
                        sys.exit('Many "access denied" answers; stopping. Re-run later.')
                else:
                    done += 1
                    total += n
                if (done + len(failed)) % 25 == 0:
                    rate = total / max(time.time() - t0, 1) / 1e6
                    left = (len(todo) - done - len(failed)) * (total / max(done, 1)) / max(rate * 1e6, 1) / 60
                    print(f'\r{done}/{len(todo)} done, {len(failed)} failed, {total / 1e9:.2f} GB, '
                          f'{rate:.1f} MB/s, about {left:.0f} min left   ', end='', flush=True)
    except KeyboardInterrupt:
        print('\nstopping (completed files are kept; re-run to continue)')
        import os
        os._exit(1)
    print(f'\nfinished: {done} downloaded, {len(failed)} failed, {total / 1e9:.2f} GB')
    if failed:
        print('Re-run the same command to retry the failed files. First error:', failed[0])


if __name__ == '__main__':
    main()
