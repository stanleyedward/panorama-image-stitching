``` sh
____                                             
|  _ \ __ _ _ __   ___  _ __ __ _ _ __ ___   __ _ 
| |_) / _` | '_ \ / _ \| '__/ _` | '_ ` _ \ / _` |
|  __/ (_| | | | | (_) | | | (_| | | | | | | (_| |
|_|   \__,_|_| |_|\___/|_|  \__,_|_| |_| |_|\__,_|

```

Panoramic image stitching with overlapping images using SIFT detector, Homography, RANSAC algorithm and blending.

## Try it yourself


1. #### Clone and cd into the repository:

    ```sh
    $ git clone https://github.com/stanleyedward/panorama-image-stitching.git
    $ cd panorama-image-stitching
     ```


2. #### Create and activate the conda environment:

    ```sh
    $ conda env create -f environment.yml
    $ conda activate panorama
    ```


3. #### Add your custom images to the `inputs/` folder manually or using the cmdline:
    ```sh
    $ mv left.jpg middle.jpg right.jpg inputs/
    ```
    dont have any images?   try the preloaded ones!


4. #### Run the script

    ```sh
    $ python panorama.py inputs/left.jpg inputs/middle.jpg inputs/right.jpg
    ```

    `Caution:` the order of the images should be `left to right` from the viewing point. 


5. #### Check it out!
    
    the output should be exported at `outputs/paranorama_image.jpg`

    ![Alt text](outputs/panorama_image.jpg)

    This is the output of the following command:

    ```sh
    $ python panorama.py inputs/back/back_01.jpeg inputs/back/back_02.jpeg inputs/back/back_03.jpeg
    ```

    ``` sh
    ____                                             
    |  _ \ __ _ _ __   ___  _ __ __ _ _ __ ___   __ _ 
    | |_) / _` | '_ \ / _ \| '__/ _` | '_ ` _ \ / _` |
    |  __/ (_| | | | | (_) | | | (_| | | | | | | (_| |
    |_|   \__,_|_| |_|\___/|_|  \__,_|_| |_| |_|\__,_|


    Initializing...
    Panoramic image saved at: outputs/panorama_image.jpg
    ```

## Explained


1. #### Feature Detection using SIFT 

The scale-invariant feature transform is a computer vision algorithm to detect interest points, describe, and match local features in images, invented by David Lowe in 1999.

![Alt text](images/sift_features_located.jpeg)

2. #### Matching keypoints
![Alt text](images/keypoints_matched.jpeg)

3. #### Computing the Homography Matrix

4. #### RANSAC algorithm

5. #### Weighted Blending
![Alt text](images/unblended_and_unsmoothed_output.jpeg)

![Alt text](images/output.png)

## References


- [First Principles of Computer Vision - Shree K. Nayar](https://fpcv.cs.columbia.edu/)
- [Distinctive Image Features from Scale-Invariant Keypoints (SIFT)](https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
- https://github.com/linrl3/Image-Stitching-OpenCV
- https://github.com/Yunyung/Automatic-Panoramic-Image-Stitching
- https://www.csie.ntu.edu.tw/~cyy/courses/vfx/12spring/lectures/handouts/lec04_stitching_4up.pdf