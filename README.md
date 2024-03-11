# Stereo Disparity: Semi-Global-Matching and Normalized Cross-Correlation

**Stereo matching** is the problem of finding correspondences between two images that are taken simultaneously from two cameras that are mounted so that they are parallel and separated along their x-axis.
The output of stereo matching is a disparity image that, for every pixel in the left image (x), indicates how many pixels to the left its correspondence (x’) is in the right image, giving the disparity (x-x’).

The primary objective of this project is to address the stereo disparity problem by integrating the **Normalized Cross-Correlation (NCC)** algorithm with the **Semi-Global Matching (SGM)** algorithm.
In addition to this integration, this project focuses on optimising the performance of these algorithms.

One aspect of optimisation involves re-formulating the NCC function for improved efficiency. Furthermore, I implement caching mechanisms to store reused data, thereby reducing computational overhead. For accelerating the SGM algorithm, I employ tensor operations in NumPy rather than conventional pixel-by-pixel iteration. This approach significantly enhances the computational efficiency of the algorithm.

## Results

<table>
    <thead>
        <tr>
            <th></th>
            <th text-align="center">Left Image</th>
            <th>Right Image</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Original</th>
            <td rowspan=1, text-align="center">
                <img src="./assets/2018-07-09-16-11-56_2018-07-09-16-13-38-366-left.jpg", width="300">
            </td>
            <td rowspan=1>
                <img src="./assets/2018-07-09-16-11-56_2018-07-09-16-13-38-366-right.jpg", width="300">
            </td>
        </tr>
        <tr>
            <th>NCC Disparity</th> 
            <td colspan=2, align="center"><img src="./assets/2018-07-09-16-11-56_2018-07-09-16-13-38-366NCC-disparity.png", width="400"></td>
        </tr>
        <tr>
            <th>SGM Disparity</th> 
            <td colspan=2, align="center"><img src="./assets/2018-07-09-16-11-56_2018-07-09-16-13-38-366SGM-disparity.png" width="400"></td>
        </tr>
    </tbody>
</table>

## Runtime Improvement

|         | Standard NCC | Accelerated NCC |
| ------- | ------------ | --------------- |
| Runtime | 517.14 (s)   | 255.15 (s)      |

## Usage

The main code is in [semi_global_match.ipynb](./semi_global_match.ipynb)

## Dependencies

The project uses `python==3.12.2`, and the dependencies can be installed by running:

```
pip install -r requirements.txt
```

## Credits

- Yu Hsien Chiang, Thomas Chen

## References

- [Stereo Processing by Semiglobal Matching and Mutual Information](https://ieeexplore.ieee.org/document/4359315)
- [Stereo Disparity Using Semi-Global Block Matching](https://www.mathworks.com/help/visionhdl/ug/stereoscopic-disparity.html)
- [Real-Time Semi-Global Matching Using CUDA Implementation
  ](https://web.stanford.edu/class/cs231a/prev_projects_2016/semi_global_cs231.pdf)
- [Template Matching using Fast Normalized Cross Correlation](https://isas.iar.kit.edu/pdf/SPIE01_BriechleHanebeck_CrossCorr.pdf)
