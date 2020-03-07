# modules used in your code
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt


def shortest_path(costs):
    """
    This function takes an array of costs and finds a shortest path from the
    top to the bottom of the array which minimizes the total costs along the
    path. The path should not shift more than 1 location left or right between
    successive rows.

    In other words, the returned path is one that minimizes the total cost:

        total_cost = costs[0,path[0]] + costs[1,path[1]] + costs[2,path[2]] + ...

    subject to the constraint that:

        abs(path[i]-path[i+1])<=1

    Parameters
    ----------
    costs : 2D float array of shape HxW
         An array of cost values

    Returns
    -------
    path : 1D array of length H
        indices of a vertical path.  path[i] contains the column index of
        the path for each row i.
    """

    path = []
    height, width = costs.shape
    M = np.zeros((height, width))

    # No previous cost to take into consideration
    for i in range(0, width):
        M[0, i] = costs[0, i]

    # for loop to compute the costs and memoize them
    for i in range(1, height):
        for j in range(0, width):
            # Left edge so can't look left, only straight and right
            if j == 0:
                M[i, j] = min(M[i - 1, j], M[i - 1, j + 1]) + costs[i, j]
            # Right edge so can't look right, only straight and left
            elif j == width - 1:
                M[i, j] = min(M[i - 1, j], M[i - 1, j - 1]) + costs[i, j]
            # Normal case where can look left, straight, and right
            else:
                M[i, j] = min(M[i - 1, j - 1], M[i - 1, j],
                              M[i - 1, j + 1]) + costs[i, j]

                # Get min at bottom row to start the backtracking
    lastRow = np.argmin(M[height - 1])
    path.append(lastRow)

    # Back track from the bottom finding the min within one away
    for i in range(1, height):
        # Left edge so can't look left, only straight and right
        if lastRow == 0:
            if (M[height - i - 1, lastRow] < M[height - i - 1, lastRow + 1]):
                next = 0
            else:
                next = 1
            lastRow = lastRow + next
            path.append(lastRow)

        # Right edge so can't look right, only straight and left
        elif lastRow == width - 1:
            if (M[height - i - 1, lastRow - 1] < M[height - i - 1, lastRow]):
                next = 0
            else:
                next = 1
            lastRow = lastRow + next - 1
            path.append(lastRow)

        # Normal case where can look left, straight, and right
        else:
            next = np.argmin(
                [(M[height - i - 1, lastRow - 1], M[height - i - 1, lastRow], M[height - i - 1, lastRow + 1])])
            lastRow = lastRow + next - 1
            path.append(lastRow)

    path = np.asarray(list(reversed(path)))
    return path


def stitch(left_image, right_image, overlap):
    """
    This function takes a pair of images with a specified overlap and stitches them
    togther by finding a minimal cost seam in the overlap region.

    Parameters
    ----------
    left_image : 2D float array of shape HxW1
        Left image to stitch

    right_image : 2D float array of shape HxW2
        Right image to stitch

    overlap : int
        Width of the overlap zone between left and right image

    Returns
    -------
    stitched : 2D float array of shape Hx(W1+W2-overlap)
        The resulting stitched image
    """

    # inputs should be the same height
    assert (left_image.shape[0] == right_image.shape[0])

    # Stiches together along a seam where the images have similar brightness values

    # 1) Extract the overlapping strips from the two input images and then compute a cost array
    #    a) cost = absolute val(left_img - right_img)
    # 2) shortest_path(cost)
    # 3) Use pixels from the left image on the left side of the seam and right images on right side of seam

    # Turn the path into an alpha mask for each image and then use the standard equation for compositing

    lHeight = left_image.shape[0]
    lWidth = left_image.shape[1]
    rHeight = right_image.shape[0]
    rWidth = right_image.shape[1]

    # Get pixels starting from the end of the left image and going backwards until overlap amount of pixels
    overlapingLeft = left_image[:, (lWidth - overlap):lWidth]
    # Get pixels starting from start of right image and going forwards until overlap amount of pixels
    overlapingRight = right_image[:, 0:overlap]
    overlapingTotal = np.concatenate((overlapingLeft, overlapingRight), axis=1)

    # Get pixels starting from start of left image until the overlap index
    NonOverlapingLeft = left_image[:, 0:(lWidth - overlap)]
    # Get pixels starting from the end of overlap until the end of the right image
    NonOverlapingRight = right_image[:, overlap:rWidth]
    NonOverlapingTotal = np.concatenate(
        (NonOverlapingLeft, NonOverlapingRight), axis=1)

    # seam = [2, 2, 1, 0, 1, 1, 2, 1, 0]
    Cost = abs(overlapingLeft - overlapingRight)

    seam = shortest_path(Cost)

    stitched = np.zeros((lHeight, lWidth + rWidth - overlap))

    for i in range(0, len(seam)):
        # Get the pixels from (0 - the seam) for left
        # Get the pixels from the (seam - end of overlapingPixels)
        leftPixelsFromSeam = overlapingLeft[i, 0:seam[i]]
        rightPixelsFromSeam = overlapingRight[i, seam[i]:]

        leftStiched = np.append(NonOverlapingLeft[i], leftPixelsFromSeam)
        rightStiched = np.append(rightPixelsFromSeam, NonOverlapingRight[i])
        stitched[i] = np.append(leftStiched, rightStiched)

    assert (stitched.shape[0] == left_image.shape[0])
    assert (stitched.shape[1] == (
        left_image.shape[1] + right_image.shape[1] - overlap))

    return stitched


def synth_quilt(tile_map, tiledb, tilesize, overlap):
    """
    This function takes the name of an image and quilting parameters and synthesizes a
    new texture image by stitching together sampled tiles from the source image.


    Parameters
    ----------
    tile_map : 2D array of int
        Array storing the indices of which tiles to paste down at each output location

    tiledb : list of int
        Dimensions of output in tiles,  e.g. (3,4)
        It is just the output of the provided "sample_tiles" function
        Tiles are patches sampled from the input texture which we are pasting down to generate the new output texture.

    tilesize : (int,int)
        Size of a tile in pixels

    overlap : int
        Amount of overlap between tiles

    Returns
    -------
    output : 2D float array
        The resulting synthesized texture of size
    """

    # determine output size based on overlap and tile size
    outh = (tilesize[0] - overlap) * tile_map.shape[0] + overlap
    outw = (tilesize[1] - overlap) * tile_map.shape[1] + overlap
    output = np.zeros((outh, outw))

    print("Output Size: ", outh, "x", outw)
    print("Tile Size: ", tilesize)
    #     print("Tile Map: ", tile_map.shape, "\n", tile_map)
    #     print("Tile DB: ", tiledb.shape, "\n", tiledb)

    superImage = np.zeros((tilesize[0], outw))
    for i in range(tile_map.shape[0]):
        # Go through each row and keep stitching images together
        myRow = np.zeros((tilesize))
        for j in range(tile_map.shape[1]):
            # If on the first tile in a row we don't have anything previous to stitch it with to the left,
            # so we need to set it as inital image
            if j == 0:
                tileNext = tiledb[tile_map[i, j]]
                tileImg = np.reshape(tileNext, tilesize)
                myRow = tileImg
            # Get the next tile over and stitch it together
            else:
                tileNext = tiledb[tile_map[i, j]]
                tileImg = np.reshape(tileNext, tilesize)
                myRow = stitch(myRow, tileImg, overlap)
        #         print("Done with row(", i, ")")

        # If its the first row we don't have anything to stitch it with above, so we need to set it as initial
        if i == 0:
            superImage = np.transpose(myRow)
        # Else transpose the row into a column so we can stitch it with the next row below, which itself is transposed
        else:
            myRowT = np.transpose(myRow)
            superImage = stitch(superImage, myRowT, overlap)

    # Image is sideways and needs to be transposed one last time
    superImage = np.transpose(superImage)
    output = superImage
    return output


# skimage is only needed for sample tiles code below


def sample_tiles(image, tilesize, randomize=True):
    """
    This function generates a library of tiles of a specified size from a given source image

    Parameters
    ----------
    image : float array of shape HxW
        Input image

    tilesize : (int,int)
        Dimensions of the tiles in pixels


    Returns
    -------
    tiles : float array of shape  npixels x numtiles
        The library of tiles stored as vectors where npixels is the
        product of the tile height and width
    """

    tiles = ski.util.view_as_windows(image, tilesize)
    ntiles = tiles.shape[0] * tiles.shape[1]
    npix = tiles.shape[2] * tiles.shape[3]
    assert (npix == tilesize[0] * tilesize[1])

    print("library has ntiles = ", ntiles, "each with npix = ", npix)

    tiles = tiles.reshape((ntiles, npix))

    # randomize tile order
    if randomize:
        tiles = tiles[np.random.permutation(ntiles), :]

    return tiles


def topkmatch(tilestrip, dbstrips, k):
    """
    This function finds the top k candidate matches in dbstrips that
    are most similar to the provided tile strip.

    Parameters
    ----------
    tilestrip : 1D float array of length npixels
        Grayscale values of the query strip

    dbstrips : 2D float array of size npixels x numtiles
        Array containing brightness values of numtiles strips in the database
        to match to the npixels brightness values in tilestrip

    k : int
        Number of top candidate matches to sample from

    Returns
    -------
    matches : list of ints of length k
        The indices of the k top matching tiles
    """
    assert (k > 0)
    assert (dbstrips.shape[0] > k)
    error = (dbstrips - tilestrip)
    ssd = np.sum(error * error, axis=1)
    ind = np.argsort(ssd)
    matches = ind[0:k]
    return matches


def quilt_demo(sample_image, ntilesout=(10, 20), tilesize=(30, 30), overlap=5, k=5):
    """
    This function takes the name of an image

    Parameters
    ----------
    sample_image : 2D float array
        Grayscale image containing sample texture

    ntilesout : list of int
        Dimensions of output in tiles,  e.g. (3,4)

    tilesize : int
        Size of the square tile in pixels

    overlap : int
        Amount of overlap between tiles

    k : int
        Number of top candidate matches to sample from

    Returns
    -------
    img : list of int of length K
        The resulting synthesized texture of size
    """

    # generate database of tiles from sample
    tiledb = sample_tiles(sample_image, tilesize)
    # number of tiles in the database
    nsampletiles = tiledb.shape[0]

    if (nsampletiles < k):
        print("Error: tile database is not big enough!")

    # generate indices of the different tile strips
    i, j = np.mgrid[0:tilesize[0], 0:tilesize[1]]
    top_ind = np.ravel_multi_index(np.where(i < overlap), tilesize)
    bottom_ind = np.ravel_multi_index(
        np.where(i >= tilesize[0] - overlap), tilesize)
    left_ind = np.ravel_multi_index(np.where(j < overlap), tilesize)
    right_ind = np.ravel_multi_index(
        np.where(j >= tilesize[1] - overlap), tilesize)

    # initialize an array to store which tile will be placed
    # in each location in the output image
    tile_map = np.zeros(ntilesout, 'int')

    print("Total Rows: ", ntilesout[0])
    print('row:')
    for i in range(ntilesout[0]):
        print(i)
        for j in range(ntilesout[1]):

            if (i == 0) & (j == 0):  # first row first tile
                matches = np.zeros(k)  # range(nsampletiles)

            elif (i == 0):  # first row (but not first tile)
                left_tile = tile_map[i, j - 1]
                tilestrip = tiledb[left_tile, right_ind]
                dbstrips = tiledb[:, left_ind]
                matches = topkmatch(tilestrip, dbstrips, k)

            elif (j == 0):  # first column (but not first row)
                above_tile = tile_map[i - 1, j]
                tilestrip = tiledb[above_tile, bottom_ind]
                dbstrips = tiledb[:, top_ind]
                matches = topkmatch(tilestrip, dbstrips, k)

            else:  # neigbors above and to the left
                left_tile = tile_map[i, j - 1]
                tilestrip_1 = tiledb[left_tile, right_ind]
                dbstrips_1 = tiledb[:, left_ind]
                above_tile = tile_map[i - 1, j]
                tilestrip_2 = tiledb[above_tile, bottom_ind]
                dbstrips_2 = tiledb[:, top_ind]
                # concatenate the two strips
                tilestrip = np.concatenate((tilestrip_1, tilestrip_2))
                dbstrips = np.concatenate((dbstrips_1, dbstrips_2), axis=1)
                matches = topkmatch(tilestrip, dbstrips, k)

            # choose one of the k matches at random
            tile_map[i, j] = matches[np.random.randint(0, k)]

    output = synth_quilt(tile_map, tiledb, tilesize, overlap)
    print("Output Size: ", output.shape)
    return output


# Testing the shortest path function
c1 = np.array([[1, 1, 1, 1], [1, 4, 6, 8], [8, 3, 1, 2],
               [4, 5, 7, 8], [3, 1, 1, 1], [0, 0, 2, 4]])
p1 = shortest_path(c1)

print(c1)
print(" Shortest Path: \n", p1)
print()


c2 = np.array([[1, 2, 3, 4], [5, 2, 9, 5], [8, 1, 1, 0], [3, 5, 1, 1], [6, 98, 3, 9], [
              0, 4, 8, 11], [1, 42, 12, 8], [3, 0, 7, 2], [4, 6, 9, 1], [1, 2, 0, 1]])
p2 = shortest_path(c2)

print(c2)
print(" Shortest Path: \n", p2)
print()

c3 = np.array([[1, 2, 3, 4, 5], [1, 6, 8, 0, 4], [1, 8, 9, 5, 3], [
              1, 88, 498, 39, 1], [50, 1, 44, 67, 34]])
p3 = shortest_path(c3)

print(c3)
print(" Shortest Path: \n", p3)
print()

###################################
# Testing to see if finding the shortest path seam and stitching based on that
###################################
left_img = plt.imread('rock_wall.jpg')
right_img = plt.imread('rock_wall.jpg')

if (left_img.dtype == np.uint8):
    left_img = left_img.astype(float) / 256

if (right_img.dtype == np.uint8):
    right_img = right_img.astype(float) / 256

# Convert to grayscale
left_img = (left_img[:, :, 0] + left_img[:, :, 1] + left_img[:, :, 2]) / 3
right_img = (right_img[:, :, 0] + right_img[:, :, 1] + right_img[:, :, 2]) / 3

# Test Arrays
left = np.array(
    [[3, 2, 9, 5, 10, 7, 4, 0, 10, 7, ],
     [2, 2, 5, 6, 1, 5, 9, 7, 5, 7, ],
     [8, 5, 6, 6, 9, 4, 6, 3, 8, 9, ],
     [7, 5, 6, 5, 7, 6, 1, 1, 8, 4, ],
     [8, 7, 4, 10, 6, 6, 2, 4, 8, 9, ],
     [10, 1, 8, 10, 9, 6, 9, 6, 8, 7, ],
     [6, 1, 2, 4, 3, 4, 5, 8, 7, 8, ],
     [2, 3, 3, 0, 7, 8, 4, 4, 9, 4, ],
     [7, 9, 10, 4, 5, 4, 7, 9, 1, 8, ]])

right = np.array(
    [[5, 6, 2, 5, 6, 10, 7, 7, 4, 9, ],
     [8, 3, 6, 6, 9, 1, 3, 4, 3, 7, ],
     [3, 5, 8, 5, 10, 7, 6, 8, 10, 6, ],
     [8, 1, 1, 7, 9, 3, 7, 8, 8, 5, ],
     [3, 10, 4, 5, 0, 3, 8, 1, 3, 10, ],
     [2, 0, 6, 5, 7, 3, 5, 3, 1, 3, ],
     [7, 5, 3, 9, 0, 9, 3, 3, 3, 7, ],
     [5, 2, 6, 4, 9, 2, 3, 3, 5, 3, ],
     [2, 2, 7, 5, 3, 7, 2, 4, 0, 8, ]])

overlap = 100

I = stitch(left_img, right_img, overlap)
# I = stitch(left, right, overlap)

print("Left Image")
plt.imshow(left_img, cmap=plt.cm.gray)
plt.show()

print("Right Image")
plt.imshow(right_img, cmap=plt.cm.gray)
plt.show()

print("Stitched Image")
plt.imshow(I, cmap=plt.cm.gray)
plt.show()

###########################
# Testing texture synthesis
###########################

print("Base Image")
baseRockImg = plt.imread('rock_wall.jpg')

if (baseRockImg.dtype == np.uint8):
    baseRockImg = baseRockImg.astype(float) / 256

baseRockImg = (baseRockImg[:, :, 0] + baseRockImg[:,
                                                  :, 1] + baseRockImg[:, :, 2]) / 3

plt.imshow(baseRockImg, cmap=plt.cm.gray)
plt.show()

defaultImg = quilt_demo(baseRockImg)
plt.imshow(defaultImg, cmap=plt.cm.gray)
plt.show()

increaseTileSizeImg = quilt_demo(baseRockImg, (10, 20), (50, 50))
plt.imshow(increaseTileSizeImg, cmap=plt.cm.gray)
plt.show()

decreaseOverlapImg = quilt_demo(baseRockImg, (10, 20), (30, 30), 3, 5)
plt.imshow(decreaseOverlapImg, cmap=plt.cm.gray)
plt.show()

increaseKImg = quilt_demo(baseRockImg, (10, 20), (30, 30), 5, 10)
plt.imshow(increaseKImg, cmap=plt.cm.gray)
plt.show()


##############################
# More testing of texture synthesis
##############################

print("Base Image")
camoImg = plt.imread('camo.png')

if (camoImg.dtype == np.uint8):
    camoImg = camoImg.astype(float) / 256

camoImg = (camoImg[:, :, 0] + camoImg[:, :, 1] + camoImg[:, :, 2]) / 3

plt.imshow(camoImg, cmap=plt.cm.gray)
plt.show()

defaultImg = quilt_demo(camoImg)
plt.imshow(defaultImg, cmap=plt.cm.gray)
plt.show()

##############################
# Even more testing of texture synthesis
##############################

print("Base Image")
blueTigerImg = plt.imread('bluetiger.png')

if (blueTigerImg.dtype == np.uint8):
    blueTigerImg = blueTigerImg.astype(float) / 256

blueTigerImg = (blueTigerImg[:, :, 0] +
                blueTigerImg[:, :, 1] + blueTigerImg[:, :, 2]) / 3

plt.imshow(blueTigerImg, cmap=plt.cm.gray)
plt.show()

defaultImg = quilt_demo(blueTigerImg)
plt.imshow(defaultImg, cmap=plt.cm.gray)
plt.show()
