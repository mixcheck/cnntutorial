import os
import sys
import tarfile
from six.moves import urllib
from matplotlib import pyplot as plt

def maybe_download_and_extract(data_url, dest_dir, file_path):
    """
    Download and extract the data if it doesn't already exist.
    Assumes the url is a tar-ball file.
    :param url:
        Internet URL for the tar-file to download.
        Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    :param download_dir:
        Directory where the downloaded file is saved.
        Example: "data/CIFAR-10/"
    :return:
        Nothing.
    """
    
    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    
    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(filepath):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
            
        # Download the file from the internet.
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_dir, file_path)

    if not os.path.exists(extracted_dir_path):
        # Unpack the tar-ball.
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)
        print('Extracting Finished')