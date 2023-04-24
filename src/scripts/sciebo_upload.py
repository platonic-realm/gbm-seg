import os
import argparse
import getpass
import logging
from datetime import datetime, timezone
import owncloud

# Set up logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO)

# Parse command-line arguments
parser = argparse.ArgumentParser(
        description='Syncs a local directory with a remote one on Sciebo.')
parser.add_argument('url', help='Sciebo server URL, '
                                'probably: https://uni-koeln.sciebo.de/')
parser.add_argument('username', help='Sciebo username')
parser.add_argument('local_dir', help='local directory to sync')
parser.add_argument('remote_dir', help='remote directory to sync')
args = parser.parse_args()

# Prompt for Sciebo server password without showing it on the screen
password = getpass.getpass("Enter your password: ")

# Connect to Sciebo server
oc = owncloud.Client(args.url)
oc.login(args.username, password)


# Recursive function to upload or download
# files in a directory and its subdirectories
def sync_directory(local_dir, remote_dir):
    # Get list of files in local directory with modification time
    local_files = {file: os.path.getmtime(os.path.join(local_dir, file))
                   for file in os.listdir(local_dir)
                   if os.path.isfile(os.path.join(local_dir, file))}

    # Get list of files in remote directory with modification time
    remote_files = {file.get_name(): file.get_last_modified()
                    for file in oc.list(remote_dir)}

    # Upload new or updated files
    for file, _ in local_files.items():

        # I decided to not check for the modification time, because Sciebo's
        # Files were always one hour early, leading to reuploading everything
        # local_modify_time = \
        #     datetime.fromtimestamp(mtime).astimezone(timezone.utc)
        # remote_modify_time = \
        #     remote_files[file].astimezone(timezone.utc)

        if file not in remote_files:
            logging.info("Uploading %s to %s",
                         os.path.join(local_dir, file),
                         remote_dir)
            oc.put_file(remote_dir + '/' + file,
                        os.path.join(local_dir, file))
        else:
            logging.info("Skipping %s (already up to date)", file)

    # Download updated files
    # for file, mtime in remote_files.items():
    #     if file not in local_files or\
    #             datetime.fromtimestamp(mtime) > local_files[file]:
    #         logging.info("Downloading %s from %s",
    #                      file,
    #                      remote_dir)
    #         oc.get_file(remote_dir + '/' + file,
    #                     os.path.join(local_dir, file))
    #     else:
    #         logging.info("Skipping %s (already up to date)", file)

    # Recurse into subdirectories
    for subdir in os.listdir(local_dir):
        if os.path.isdir(os.path.join(local_dir, subdir)):
            subdir_local = os.path.join(local_dir, subdir)
            subdir_remote = remote_dir + '/' + subdir
            subdir_exists = False
            for d in oc.list(remote_dir):
                if d.get_name() == subdir:
                    subdir_exists = True
                    break
            if not subdir_exists:
                logging.info("Creating directory %s", subdir_remote)
                oc.mkdir(subdir_remote)
            sync_directory(subdir_local, subdir_remote)


# Sync the top-level directory
sync_directory(args.local_dir, args.remote_dir)
