# Image Dehazing

This repo contains a python implementation of image dehazing as described in **Non-Local Image Dehazing**, by Berman et al. (CVPR 2016) [1].

Official MATLAB implementation is here: https://github.com/danaberman/non-local-dehazing

# Background

The idea behind image dehazing is to improve the visibility of images captured in hazy/foggy weather conditions. The light absorption caused by haze (dust, smoke, water droplets, etc.) reduces contrast and alters the color of objects in pictures resulting in a washed-out look.

Dehazing matters not only for outdoor photography but also for stuff like remote sensing, autonomous navigation, etc.


![alt text](/images/image.png)
Taken from [4].

Most non-ML dehazing methods are based on the atmospheric scattering hazy image formation model [2]:
$$
I(x)=t(x) \cdot J(x)+[1-t(x)] \cdot A
$$

where $I(x)$ is the observed intensity of the hazy image at pixel $x$, $J(x)$ represents the true radiance of the scene (the true colors of the object) in the absence of haze, and $A$ is the global atmospheric light (air-light) representing the intensity of the ambient light caused by scattered particles (this is tricky to estimate for dehazing).

$t(x)$ is the transmission map. It describes the portion of the light that isn't scattered and reaches the camera. It is dependent on the distance of the object from the camera:
$$
t(x)=e^{-\beta d(x)},
$$

where $\beta$ is the attenuation coefficient of the atmosphere and $d(x)$ is the distance of the scene at pixel $x$. Since $\beta$ is wavelength-dependent $t$ is different per color channel.


# Non-Local Image Dehazing

- **Input:** $\boldsymbol{I}(\boldsymbol{x}), \boldsymbol{A}$  
- **Output:** $\hat{\boldsymbol{J}}(\boldsymbol{x}), \hat{t}(\boldsymbol{x})$

1. $\boldsymbol{I}_{\mathrm{A}}(\boldsymbol{x}) = \boldsymbol{I}(\boldsymbol{x}) - \boldsymbol{A}$
2. Convert $\boldsymbol{I}_{\mathrm{A}}$ to spherical coordinates to obtain $[r(\boldsymbol{x}), \phi(\boldsymbol{x}), \theta(\boldsymbol{x})]$
3. Cluster the pixels according to $[\phi(\boldsymbol{x}), \theta(\boldsymbol{x})]$. Each cluster $H$ is a haze-line.
4. For each cluster $H$:
   - Estimate maximum radius: $\hat{r}_{\text{max}}(\boldsymbol{x}) = \max_{\boldsymbol{x} \in H}\{r(\boldsymbol{x})\}$
5. For each pixel $x$:
   - Estimate transmission: $\tilde{t}(\boldsymbol{x}) = \frac{r(\boldsymbol{x})}{\hat{r}_{\text{max}}}$
6. Perform regularization by calculating $\hat{t}(\boldsymbol{x})$ that minimizes the following equation:
   - $\sum_{\boldsymbol{x}} \frac{\left[\hat{t}(\boldsymbol{x}) - \tilde{t}_{LB}(\boldsymbol{x})\right]^2}{\sigma^2(\boldsymbol{x})} + \lambda \sum_{\boldsymbol{x}} \sum_{\boldsymbol{y} \in N_{\boldsymbol{x}}} \frac{[\hat{t}(\boldsymbol{x}) - \hat{t}(\boldsymbol{y})]^2}{\|\boldsymbol{I}(\boldsymbol{x}) - \boldsymbol{I}(\boldsymbol{y})\|^2}$
7. Calculate the dehazed image using the dehazing equation.


**Dehazing Equation** 

Once $\hat{t}(\boldsymbol{x})$ is calculated as the minimum of
$$\sum_{\boldsymbol{x}} \frac{\left[\hat{t}(\boldsymbol{x})-\tilde{t}_{L B}(\boldsymbol{x})\right]^2}{\sigma^2(\boldsymbol{x})}+\lambda \sum_{\boldsymbol{x}} \sum_{\boldsymbol{y} \in N_{\boldsymbol{x}}} \frac{[\hat{t}(\boldsymbol{x})-\hat{t}(\boldsymbol{y})]^2}{\|\boldsymbol{I}(\boldsymbol{x})-\boldsymbol{I}(\boldsymbol{y})\|^2}$$

the dehazed image is calculated using:
$$
\hat{\boldsymbol{J}}(\boldsymbol{x}) = \left\{\boldsymbol{I}(\boldsymbol{x}) - [1 - \hat{t}(\boldsymbol{x})] \boldsymbol{A}\right\} / \hat{t}(\boldsymbol{x}) .
$$

In the model, each haze-line in RGB space is defined by its relationship to the air-light value $A$ (these lines originate from the Air-light point and extend through the color space).

Estimating the Air-light is crucial because it determines the base point from which haze-lines are drawn. Better Air-light estimation means more accurate reconstruction of the transmission map and a more true-to-life dehazed image.

Air-light is estimated applying the method described in [3] which I outline below.

# Air-light Estimation

- **Input**: Hazy image $\boldsymbol{I}(\boldsymbol{x})$  
- **Output**:  $\hat{A}$
1. Cluster the pixels' colors and generate an indexed image $\hat{\boldsymbol{I}}(\boldsymbol{x})$ whose values are $n \in\{1, \ldots, N\}$, a colormap $\left\{\boldsymbol{I}_n\right\}_{n=1}^N$, and cluster sizes $\left\{w_n\right\}_{n=1}^N$
2. **for** each pair of color channels $\left(c_1, c_2\right) \quad \in$ $\{R G, G B, R B\}$ **do**
    - Initialize accum $c_{c_1, c_2}$ to zero
    - **for** $\boldsymbol{A}=(m \cdot \Delta A, l \cdot \Delta A), m, l \in\{0, \ldots, M\}$ **do**
        - **for** $\theta_k=\frac{\pi}{K}, k \in\{1, \ldots, K\}$ **do**
            - **for** $n \in\{1, \ldots, N\}$ **do**
                - $d=\left|\left(\boldsymbol{A}-\boldsymbol{I}_n\left(c_1, c_2\right)\right) \times\left(\cos \left(\theta_k\right), \sin \left(\theta_k\right)\right)\right|$
                **if** $(d<\tau) \wedge\left(m \cdot \Delta A>I_n\left(c_1\right)\right) \wedge$ $\left(l \cdot \Delta A>I_n\left(c_2\right)\right)$ **then**
                    - $\operatorname{accum}_{c_1, c_2}(k, m, l)+=w_n \cdot f\left(\left\|\boldsymbol{A}-\boldsymbol{I}_n\right\|\right)$
3. $\hat{\boldsymbol{A}}=\arg \max \left\{\operatorname{accum}_{R, G} \otimes \operatorname{accum}_{G, B} \otimes \operatorname{accum}_{R, B}\right\}$, where $\otimes$ is an outer product
4. Return

![alt text](/images/image-2.png)

The algorithm clusters pixel's colors in the hazy image and then uses a Hough Transform in 2D color space to vote for the best air-light candidate based on how well the haze-lines (groups of pixels spread out along RGB color space lines) converge at candidate points. The point of convergence is the estimated air-light.

The estimated air-light serves as input to the dehazing algorithm image formation model. It separates the light scattered by atmospheric particles from the actual light reflected by objects, and corrects the color and intensity of each pixel based on its estimated distance from the camera and the amount of scattered light.


# Testing (Jupyter Notebook)

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