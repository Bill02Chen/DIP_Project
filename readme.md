# Image Art Style Changing

## Description
A really small but interesting expirement in using tradition image processing way to change image's art style.

## Methodology
Basic idea is smoothing and edge enhancement. There are also some adaptive function designed by myself.


## How to use

First run matlab code: smooth.m
In order in run this matlab code, there is a need to install a matlab package (Image Processing Toolbox).
You need to manually modify the input image path in this matlab code file.

Then run: python3 mainpart.py input_image_path

The output image is named as output.png

I also provide some images for you to test, they are in folder test

The function L0Smoothing.m is provided by the reference paper, thanks to their contribution.

### References
Li Xu, Cewu Lu, Yi Xu, and Jiaya Jia. Image smoothing via l0 gradient minimization. In Proceedings of the 2011 SIGGRAPH Asia conference, pages 1–12, 2011. 

