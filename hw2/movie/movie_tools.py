"""
Requires the moviepy Python package.
"""
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import matplotlib.pyplot as plt

def _make_grayscale(frame) :
    return 0.299*frame[:,:,0] + 0.587*frame[:,:,1] + 0.114*frame[:,:,2]

"""
Given a movie filename, returns a numpy ndarray of the frames.
The shape is (num_frames,height,width,3) where the 3 signifies 
Red, Green, and Blue channels.  Warning: The resulting matrices 
can be very big depending on the quality and length of the movie.
"""
def load_movie(filename) :
    clip = VideoFileClip(filename)
    frameList = [_make_grayscale(a) for a in clip.iter_frames()]
    mat = np.asarray(frameList,dtype=np.uint8)
    return mat

def _truncate_image(img) :    
    result = np.empty(img.shape)
    for i in range(3) :
        m = np.amin(img[:,:,i])
        M = np.amax(img[:,:,i])
        result[:,:,i] = img[:,:,i] + m*1.0
        result[:,:,i] = result[:,:,i] / M*1.0
    return result

"""
Plots a list of vectors on the screen in a grid with n_col columns
and n_row rows.  The resulting grid is saved in the file title.pdf.
If bycol is True, then the images are laid out by columns instead of by rows.
Supports optional row_titles and column_titles given as lists of strings.
"""
def plot_image_grid(images, title, image_shape=(28,28),n_col=5, n_row=2, bycol=0, row_titles=None,col_titles=None):
    fig,axes = plt.subplots(nrows=n_row,ncols=n_col,figsize=(2. * n_col, 1.5 * n_row))
    for i, comp in enumerate(images):
        row,col = reversed(divmod(i,n_row)) if bycol else divmod(i,n_col)       
        cax = axes[row,col] if n_row > 1 else axes[col]
        cax.imshow(comp.reshape(image_shape),
                   interpolation='nearest',cmap=plt.cm.gray,
                   vmin=comp.min(), vmax=comp.max())
        cax.set_xticks(())
        cax.set_yticks(())
    if row_titles is not None :
        axli = axes if n_col == 1 else axes[:,0]
        for ax,row in zip(axes[:,0],row_titles) :
            ax.set_ylabel(row,size='large')
    if col_titles is not None :
        axli = axes if n_row == 1 else axes[0]
        for ax,col in zip(axli,col_titles) :
            ax.set_title(col)
    
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.tight_layout()
    plt.savefig(title + '.pdf',bbox_inches='tight')
    plt.show()

def plot_image(image, title, image_shape=(28,28)) :
    plt.imshow(image.reshape(image_shape),
               interpolation='nearest',cmap=plt.cm.gray,
                vmin=image.min(), vmax=image.max())
    plt.xticks(())
    plt.yticks(())
    plt.savefig(title + '.pdf',bbox_inches='tight')

