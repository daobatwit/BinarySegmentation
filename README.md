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
Packages: keras, ssl, tensorflow, segmentation_models

 * Configuration <br />
Edit config.py to your file explorer path specifications

 * Dataset <br />
The dataset used to test image segmentation is stored as a zip file in this link below:

https://drive.google.com/file/d/1x5IJKjV9dXxixv-R_dNZdxpA1jcla1st/view?usp=sharing

Inside of the zip file contains all the images used for training/validation. Each model's training and validation set are also in these folders. The fixed test set used to
test each model is in this folder as well.

 * Installation <br />
1. Clone repo
2. Download data sets
3. Update paths in both config.py and util.py
4. Update paths in manualSegment.ipynb file
5. Run manualSegment.ipynb to obtain images for seg.ipynb script
6. Run seg.ipynb

For Iteration Training Stragety:
We used apeer.com to do the manual segmentation afterwards to add from the failed test result to the training set.
Needed applications: ImageJ and an account for apeer.com
https://imagej.nih.gov/ij/download.html

Steps to manually segment:
1. Create an account in apeer.com
2. Go to annotate tab and create a new dataset
3. Once created, click on the new dataset and import the images needed for annotation
4. When annotating, a new window to create the masks.
5. Create a new class different from background to annotate the masks
6. Select the new class, and select the brush tool to outline the joints
7. Once the joints are highlighted, select the export button
8. When exporting, select the class which contains the highlighted mask and download the file
9. The file is saved as a tiff image, so we need to convert it from a tiff to png
10. Use the link provided above to download the application ImageJ
11. Open ImageJ, and open the downloaded file
12. Once opened, select the Image tab
13. Under the Image tab, select the adjust option and select Brightness/Contrast
14. Once opened, adjust the Brightness from 255 to 1.
15. The Image should change from a black image to the joint mask.
16. Save the image as a png file
