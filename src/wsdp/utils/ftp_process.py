import os
import ftplib

from tqdm import tqdm
from urllib.parse import urlparse, unquote
from .load_preset import load_api


def download_ftp(dataset_name: str, dest: str, extensions: list = None):
    """
    Download a dataset via FTP with optional extension filtering.

    Args:
        dataset_name: Name of the dataset (used to look up FTP URL from api.json)
        dest: Destination directory
        extensions: If provided, only download files with these extensions.
                    e.g. ['.csv', '.mat'] to skip .dat binary files.
                    None means download everything.
    """
    url = load_api(dataset_name)
    parsed = urlparse(url)
    ftp_server = parsed.hostname
    ftp_port = parsed.port
    ftp_user = parsed.username
    ftp_pass = parsed.password
    ftp_root_path = unquote(parsed.path)

    try:
        ftp = ftplib.FTP()
        ftp.connect(ftp_server, ftp_port)
        ftp.login(ftp_user, ftp_pass)

        ftp.encoding = 'utf-8'
        ftp.set_pasv(True)
        print(f"prepare to download from: {ftp_root_path}")
        if extensions:
            print(f"extension filter: {extensions} (other formats will be skipped)")

        try:
            ftp.cwd(ftp_root_path)
        except ftplib.error_perm as e:
            print(f"Error: cannot dive into: '{ftp_root_path}'.")
            raise e

        stats = {'downloaded': 0, 'skipped': 0}
        _download_current_dir(ftp, dest, extensions, stats)

        ftp.quit()
        print(f"\ndownload complete: {stats['downloaded']} files downloaded, {stats['skipped']} files skipped")

    except Exception as e:
        raise e


def _download_current_dir(ftp, local_dir, extensions=None, stats=None):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    if stats is None:
        stats = {'downloaded': 0, 'skipped': 0}

    try:
        filenames = ftp.nlst()
    except ftplib.error_perm:
        return

    for filename in filenames:
        if filename in ('.', '..'):
            continue

        local_path = os.path.join(local_dir, filename)

        try:
            ftp.cwd(filename)
            _download_current_dir(ftp, local_path, extensions, stats)
            ftp.cwd('..')

        except ftplib.error_perm:
            # Extension filter: skip files that don't match
            if extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext.lower() not in [e.lower() for e in extensions]:
                    stats['skipped'] += 1
                    continue

            try:
                try:
                    file_size = ftp.size(filename)
                except:
                    file_size = None

                with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename, leave=False) as pbar:
                    with open(local_path, 'wb') as f:
                        def download_callback(data):
                            f.write(data)
                            pbar.update(len(data))

                        ftp.retrbinary('RETR ' + filename, download_callback)

                stats['downloaded'] += 1

            except Exception as e:
                raise e
