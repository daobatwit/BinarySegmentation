# JSW Image Segmentation
The aim of this project is to RBG take X-ray images and turn them into segmented binary images with only the two segments of a joint included at the end. We train a U-net model
to accomplish this after manually segmenting images and creating their associated masks. 

CONTENTS OF THIS FILE
---------------------

 * Introduction <br />
Osteoarthritis is the most common form of arthritis, affecting roughly 32 million people in the US according to the CDC. With these processed, binary joint images, a future
project can use them to measure the space between joints to effectively measure the space between joints to diagnose the stage of Osteoarthritis. 

* Requirements <br />
Python version 3.9.7 or later <br />
Packages: keras, ssl

 * Configuration <br />
Edit config.py to your file explorer path specifications

 * Installation <br />
1. Clone repo
2. Download data sets
3. run manualSegment.ipynb to obtain images for seg.ipynb script
4. run seg.ipynb


