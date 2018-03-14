"""
Poll the GPU server to download model updates via FTP
"""
import json
import paramiko
from time import sleep


def poll_ftp(settings, sleep_time=300):

    remote_path = '/home/ubuntu/data/chess-deep-learning/best_model.hdf5'
    local_path = '/Users/tom/tmp/best_model.hdf5'

    last_mod = 0

    while True:
        transport = paramiko.Transport((settings['host'], settings['port']))
        pkey = paramiko.RSAKey.from_private_key_file(settings['key_path'])
        
        transport.connect(username=settings['user'], pkey=pkey)
        sftp = paramiko.SFTPClient.from_transport(transport)

        stats = sftp.stat(remote_path)
        if stats.st_mtime != last_mod:
            print(f'Modified, downloading to {local_path}')
            last_mod = stats.st_mtime
            sftp.get(remote_path, local_path)
        else:
            print('Not Modified')

        sftp.close()
        transport.close()

        sleep(sleep_time)


if __name__ == '__main__':
    settings = json.load(open('/Users/tom/Projects/Portfolio/data/chess-deep-learning/ftp_settings.json'))
    poll_ftp(settings)
