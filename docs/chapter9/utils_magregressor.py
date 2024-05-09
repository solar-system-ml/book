import numpy as np


def rotateFrames(frames):
    print(frames.shape)
    return np.swapaxes(frames, 2, 3)

def reshape_to_3d(frames):
    (A, B, C) = frames.shape

    n_im = int(C/B)

    output = np.zeros((A, n_im, B, B), dtype=frames.dtype)
    for i in range(A):
        for j in range(n_im):
            output[i, j, :, :] = frames[i, :, j*B:(j+1)*B]
    return output

def make_display_image(frame, step=9):
    (A, B, C) = frame.shape

    for i in range(0,int(A/3),step):
        if i ==0:
            im = frame[i, :, :].T
        else:
            im = np.concatenate([im,frame[i, :, :].T])
    return im.T


def shuffle(frame, expandFactor=3, n_shuffle = 2):
    """
    shuffle around the image by small amount

    Frame is a 3d frame of shape (C, A, A)

    the shuffling will be +/- 1/expandFactor pixels in both x and y

    """

    r = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
    (A,B,C) = frame.shape

    shuffled_frames = []
    for j in range(n_shuffle):
        shuffled_frames.append([])
        x = (np.random.rand(n_shuffle)*8).astype('int')
        for i in range(A):

            expanded = expand2d(frame[i, :, :], int(expandFactor))

            if r[x[j]][0]<0:
                expanded[:, :-1] = expanded[:, 1:]
            elif r[x[j]][0]>0:
                expanded[:, 1:] = expanded[:, :-1]
            if r[x[j]][1]<0:
                expanded[:-1, :] = expanded[1:, :]
            elif r[x[j]][1]>0:
                expanded[1:, :] = expanded[:-1, :]
            shuffled_frames[j].append(downSample2d(expanded, int(expandFactor)))

    return shuffled_frames

def shuffle_all(frames, labels, expandFactor=3, n_shuffle=1):
    new_frames = []
    new_labels = []
    for i in range(int(len(frames)/2)):
        if (i+1)%100==0:
            print(i+1,len(frames))
        shuffled_frames = shuffle(frames[i],expandFactor=3,n_shuffle=1)
        for j in range(len(shuffled_frames)):
            new_frames.append(shuffled_frames[j])
            new_labels.append(labels[i])

    return (np.array(new_frames), np.array(new_labels))
