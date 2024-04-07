# Image Dehazing

This repo contains a python implementation of image dehazing as described in **Non-Local Image Dehazing**, by Berman et al. (CVPR 2016) [1].

Official MATLAB implementation is here: https://github.com/danaberman/non-local-dehazing

# Table of Contents

- [Background](#background)
- [Non-Local Image Dehazing](#non-local-image-dehazing)
- [Air-light Estimation](#air-light-estimation)
- [Testing - Results](#testing---results)
- [TODO](#todo)
- [References](#references)

# Background

The idea behind image dehazing is to improve the visibility of images captured in hazy/foggy weather conditions. The light absorption caused by haze (dust, smoke, water droplets, etc.) reduces contrast and alters the color of objects in pictures resulting in a washed-out look.

Dehazing matters not only for outdoor photography but also for stuff like remote sensing, autonomous navigation, etc.


![alt text](/images/image.png)
Taken from [4].

Most non-ML dehazing methods are based on the atmospheric scattering hazy image formation model [2]:

$$I(x)=t(x) \cdot J(x)+[1-t(x)] \cdot A$$

where $I(x)$ is the observed intensity of the hazy image at pixel $x$, $J(x)$ represents the true radiance of the scene (the true colors of the object) in the absence of haze, and $A$ is the global atmospheric light (air-light) representing the intensity of the ambient light caused by scattered particles (this is tricky to estimate for dehazing).

$t(x)$ is the transmission map. It describes the portion of the light that isn't scattered and reaches the camera. It is dependent on the distance of the object from the camera:

$$t(x)=e^{-\beta d(x)}$$

where $\beta$ is the attenuation coefficient of the atmosphere and $d(x)$ is the distance of the scene at pixel $x$. Since $\beta$ is wavelength-dependent $t$ is different per color channel.


# Non-Local Image Dehazing

![alt text](/images/eq1.png)

![alt text](/images/eq2.png)

The model is based on the concept of haze-lines in RGB space, which are formed by haze affecting different pixels at different distances from the camera.   
The idea is that colors in a haze-free image can be approximated by a few distinct clusters in RGB space which when distorted by haze, form these haze-lines. The algorithm estimates the transmission map and air-light for each pixel using the haze-lines to recover the clear image.

Each haze-line in RGB space is defined by its relationship to the air-light value $A$ (these lines originate from the Air-light point and extend through the color space). 
Estimating the Air-light is crucial because it determines the base point from which haze-lines are drawn. Better Air-light estimation means more accurate reconstruction of the transmission map and a more true-to-life dehazed image.

Air-light is estimated applying the method described in [3] which I outline below.

# Air-light Estimation

![alt text](/images/eq3.png)

![alt text](/images/image-2.png)

The algorithm clusters pixel's colors in the hazy image and then uses a Hough Transform in 2D color space to vote for the best air-light candidate based on how well the haze-lines (groups of pixels spread out along RGB color space lines) converge at candidate points. The point of convergence is the estimated air-light.

The estimated air-light serves as input to the dehazing algorithm image formation model. It separates the light scattered by atmospheric particles from the actual light reflected by objects, and corrects the color and intensity of each pixel based on its estimated distance from the camera and the amount of scattered light.


# Testing - Results

![alt text](/images/image-9.png)
![alt text](/images/image-10.png)
![alt text](/images/image-11.png)
![alt text](/images/image-14.png)
![alt text](/images/image-12.png)
![alt text](/images/image-13.png)
![alt text](/images/image-15.png)
![alt text](/images/image-1.png)


# TODO
- [x] Air light estimation
- [ ] Radiometric correction experiments
- [ ] Test on different datasets (indoor, outdoor, remote sensing, etc.)
- [ ] Comparison with ML models
- [ ] Objective quality measurements (PSNR, SSIM, GMSD, ETC.)

# References

[1] Berman, D., Treibitz, T., & Avidan, S. Non-Local Image Dehazing. IEEE Conf. CVPR 2016.

[2] Mccartney, E.J.; Hall, F.F. Optics of the Atmosphere: Scattering by Molecules and Particles. Phys. Today 1977, 30, 76–77. https://pubs.aip.org/physicstoday/article-abstract/30/5/76/431713/Optics-of-the-Atmosphere-Scattering-by-Molecules?redirectedFrom=fulltext

[3] Berman, D., Treibitz, T., & Avidan, S. Air-light Estimation Using Haze-Lines. IEEE Conf. ICCP 2017.

[4] Ancuti, C. O., Ancuti, C., Timofte, R., & Vleeschouwer, C. D. I-HAZE: a dehazing benchmark with real hazy and haze-free indoor images. ArXiv:1804.05091v1. 2018
