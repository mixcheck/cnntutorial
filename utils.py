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

def plot_images(images, true_class, class_names, prediction=None, smooth=True):
    """Plot 9 images. 
    This function is from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/06_CIFAR-10.ipynb
    """
    assert len(images) == len(true_class) == 9
    
    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)
    # Adjust vertical spacing if we need to print ensemble and best-net.
    if prediction is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        
        # Plot image.
        ax.imshow(images[i, :, :, :],
                interpolation=interpolation)
        
        # Name of the true class.
        cls_true_name = class_names[true_class[i]]
    
        # Show true and predicted classes.
        if prediction is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[prediction[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
    
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    #plt.show()
    return fig