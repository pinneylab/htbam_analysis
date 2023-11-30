# Image stitching with [ImageStitcher](https://github.com/FordyceLab/ImageStitcher)
**Fordyce Lab, 2020**
<br>
**Author: Daniel Mokhtari**


## Purpose
[ImageStitcher](https://github.com/FordyceLab/ImageStitcher) is a simple package for flat-field correcting, stitching, and background subtracting rastered image tiles acquired via [MicroManager](https://micro-manager.org/) or [RunPack](https://github.com/FordyceLab/RunPack).


## Installation
This guide assumes that you have (1) set up a anaconda virtual environment with Python ≥ 3.6), (2) installed the iPython kernel and Jupyter, and (3) registered your environment with your installation of jupyter. Instructions on how to do this can be found [here](https://www.google.com/search?q=make+conda+environment&oq=make+conda+environment), [here](https://www.google.com/search?q=conda+install+ipython+kernel), and [here](https://www.google.com/search?q=register+ipython+kernel+jupyter).

### i. Activate a Conda virtual environment configured with an iPython kernel registered with Jupyter in a terminal session

### ii. PIP install ImageStitcher
1. Download the [ImageStitcher](https://github.com/FordyceLab/ImageStitcher) zip (private) from the FordyceLab Github
2. Change directory to unzipped package path
	- `$ cd /repo-download-dir`
3. PIP install the package in place and make editable
	- `$ pip install -e .`

### iii. Launch the [example notebook](notebooks/basic_stitcher.ipynb) from your Jupyter session

<br>

# Usage

## i. Define stitching settings
First, we need to define parameters to desribe what the images look like, how the raster was acquired (which corner of the rastered region did the imaging start at?), and load any necessary flat-field correction images and parameters. Reference free flat-field corrections are also possible, but not implemented here.<sup>1</sup>

**a. Define any needed flat-field images and parameters**

```python
# Flat field image path
setup_eGFP_ffPath = '/Setup2_FF_eGFP_500ms_2x2.tif'

# Flat-field fit parameters {channel_1: {exposure_t_1: (D, m), exposure_t_2: (D, m), ...},
#			     channel_2: {exposure_t_1: (D, m), exposure_t_2: (D, m), ...}
#			     }
# With empirically fit parameters
# D = fit dark field value (flat)
# m = image average of F-D where F = flat-field image

setup_ffParams = {'4egfp': {500: (-150, 16665)}}
```
Note:

* Flat-field image must be the same dimension (binning) as target images to be corrected
* Exposure times for FF parameters are in *ms*, and must match the target images

<br>

**b. Instantiate a StitchingSettings object**

```python
settings = stitcher.StitchingSettings(ffPaths = {'4egfp': setup_eGFP_ffPath},
                                        ffParams = setup_ffParams,
                                        setupNum= 2,
                                        tileDim = 1024
                                     )
                                     
# Or if no flat-field images/parameters
settings = stitcher.StitchingSettings(setupNum= 2, tileDim = 1024)

```
Note:

* The setupNum defines the raster origin and pattern, as this differs among setups
* tileDim is the width or height of the target image. ImageStitcher assumes that the images are 

<br>

## ii. Stitch images

### a. RunPack imaging
RunPack derived images are flat rasters (not stacked) with embedded metadata including acquisition time, raster position, etc.. These images are either collected as single scans ("single"), or kinetic time series ("kinetic"). ImageStitcher treats these classes of imaging slightly differently when stitching (saves the images at different levels of the hierarchy) to facilitate pipelining in downstream applications. The structure of these images, common to both scans and kinetic acquisitions, is shown below

**Structure**

<pre>
<code>parent-root
+-- channel-1
»	+-- YYYYmmdd-HHMMSS_Description_channel-1
»		+-- 1-Pos_XXX_YYY_ExposureT-1.tif
»		+-- 1-Pos_XXX_YYY_ExposureT-1.tif
»		+-- ...
»		+-- 1-Pos_XXX_YYY_ExposureT-2.tif
»		+-- 1-Pos_XXX_YYY_ExposureT-2.tif
»		+-- ...
»	+-- YYYYmmdd-HHMMSS_Description_channel-1
»		+-- 1-Pos_XXX_YYY_ExposureT-1.tif
»		+-- 1-Pos_XXX_YYY_ExposureT-1.tif
»		+-- ...
»		+-- 1-Pos_XXX_YYY_ExposureT-2.tif
»		+-- 1-Pos_XXX_YYY_ExposureT-2.tif
»		+-- ...
»		...
+-- channel-2
»	+-- YYYYmmdd-HHMMSS_Description_channel-2
»		+-- 1-Pos_XXX_YYY_ExposureT-1.tif
»		+-- 1-Pos_XXX_YYY_ExposureT-1.tif
»		+-- ...
»		+-- 1-Pos_XXX_YYY_ExposureT-2.tif
»		+-- 1-Pos_XXX_YYY_ExposureT-2.tif
»		+-- ...
»	+-- YYYYmmdd-HHMMSS_Description_channel-2
»		+-- 1-Pos_XXX_YYY_ExposureT-1.tif
»		+-- 1-Pos_XXX_YYY_ExposureT-1.tif
»		+-- ...
»		...
...</code>
</pre>


**Usage**

```python
multiImagePath = '/parent-root' #or higher
overlap = 0.1

p = stitcher.RasterParams(overlap, autoFF = True)

stitcher.walkAndStitch(multiImagePath, p, stitchtype = 'kinetic')
# Alternate stitchtype = 'single'

```

**multiImagePath**: (str | pathlib.Path | path-like object) path to the root of the imaging directory structure

**overlap**: (float) fractional overlap of tiled images [0–1)

**autoFF**: (bool) flag for automatically applying flat-field corrections, as defined in the stitcher.StitchingSettings

**stitchtype**: (str: "single" | "kinetic") type of stitching to perform (dictates stitched image out path). If stitching a time series, use "kinetic". Else, use "single".


<br>


### b. Micro-Manager .ome.tif stack imaging
Micro-Manager can export and convert between flat and stacked images. For simplicity, ImageStitcher only supports stitching the stacked .ome.tif Micro-Manager images. If you collected flat rasters, convert them to stacks using Micro-Manager before proceeding.

**Structure (all exposure times and channels are stacked)**

<pre>
<code>parent-root
+-- Description_1_MMStack_1-Pos_XXX_YYY.ome.tif
+-- Description_1_MMStack_1-Pos_XXX_YYY.ome.tif
...</code>
</pre>


**Usage**

```python
root = '/stack-parent'
overlap = 0.1

 # Channel names from MicroManager configuration
channelExposureMap = {'3-GFP-B': 500, '5------': 100}

# Remap the names of the channels for saved filenames, if desired
channelRemap = {'3-GFP-B': '3-GFP-B', '5------': 'Cy5'}

p = stitcher.RasterParams(overlap, autoFF = False)
stitcher.MMStitchStacks(root,
                        p,
                        channelExposureMap, 
                        channelRemap = channelRemap
                        )
```

**root**: (str | pathlib.Path | path-like object) root of the image stacks

**overlap**: (float) fractional overlap of tiled images [0–1)

**channelExposureMap**: (dict) Micro-Manager metadata doesn't retain exposure times, so you need to specify these as a dictionary. Note that Micro-Manager also doesn't permit the same channel name to have more than one exposure time per stack, giving rise to the dict structure shown.

**channelRemap**: (dict) Micro-Manager default channel names may not be descriptive for naming purposes, so remap them if required.

**autoFF**: (bool) flag for automatically applying flat-field corrections, as defined in the stitcher.StitchingSettings

<br>

### c. Image background subtraction

**Description:** In ImageStitcher, background subtraction is performed on the full stitched images. This is easily done by 1) storing the reference/background images and their channels/exposures, and 2) specifying the highest level of the directory structure to walk down and perform background subtractions of all stitched images of corresponding parameters


*Set up reference images*

```python
bg = stitcher.BackgroundImages()

# Reference background images
backgroundsRootD1 = 'd1-parent/StitchedImg_500_4egfp_0.tif'
backgroundsRootD2 = 'd2-parent/StitchedImg_500_4egfp_0.tif'

device_1 = 'd1'
device_2 = 'd2'
reference_channel = '4egfp'
reference_exposure = 500 #ms
bg.add(backgroundsRootD1, device_1, reference_channel, exposure)
bg.add(backgroundsRootD1, device_2, reference_channel, exposure)
```

*Execute*

```python
target = '/root-path'
target_device = 'd1'
target_channel = '4egfp'
bg.walkAndBGSubtract(targetRoot, target_device, channel)
```

___
<sup>1</sup> Peng, T., Thorn, K., Schroeder, T. *et al.* A BaSiC tool for background and shading correction of optical microscopy images. *Nat Commun* **8**, 14836 (2017) doi:10.1038/ncomms14836
