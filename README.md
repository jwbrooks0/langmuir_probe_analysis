# langmuir_probe_analysis
Single Langmuir Probe Analysis


This code implements a single Langmuir Probe analysis and is primarily based on the work in reference [1].


### Example results

![Example 1](https://github.com/jwbrooks0/langmuir_probe_analysis/blob/main/Example_1_results.png "Example 1")

![Example 2](https://github.com/jwbrooks0/langmuir_probe_analysis/blob/main/Example_2_results.png "Example 2")

![Example 3](https://github.com/jwbrooks0/langmuir_probe_analysis/blob/main/Example_3_results.png "Example 3")



### Notes
* This work is provided "as is" with no gaurantees as to its accuracy or the presence of bugs.  
* There are a lot of "nuances" associated with this code and Langmuir probe analysis in general.  I.e. this code can be confusing, make assumptions invalid for your application, there are many "conventions" concerning Langmuir probe analysis, etc.  
* PLEASE read through all references and code's docstring first before submitting any questions to me.  This being said, please let me know if you discover any bugs or have other suggestions for improvements.  
* This code (at present) does not calculate the final ion density as discussed in [1].  
* V_plasma can be calculated in two ways in this code.  First is using the derivative of a smoothed electron current.  Note that this method does not always work particularly if SNR is low, there aren't enough points, and if the sheath is 'thick'.  The second method uses the definition of V_plasma in Lamguir probe theory.  While potentially less accurate, this method is much more robust. 
* If you need to debug the code (or debug how the code processes your data), set vebose=True and plot_misc=True.   
	 
### FAQ

__Why doesn't this code work on my data?__

There are MANY possibilities.  First, check my notes above about nuances, assumptions, conventions, etc.  Second: you may have too few points between V_float and V_plasma.  Retake your data but with more points.  Third: Your data has low SNR.  Try filtering first.  Or retake your data but with more averaging.  Fourth:  Your sheath must be "thin" for this code to work.  I.e. your probe radius should ideally be at least 10 times smaller than the debye length.  Increase the size of your probe and take more data.
	 
		

### References
* [1] Lobbia and Beal, "Recommended Practice for Use of Langmuir Probes in Electric Propulsion Testing" https://doi.org/10.2514/1.B35531
* [2] Merlino, "Understanding Langmuir probe current-voltage characteristics" https://aapt.scitation.org/doi/10.1119/1.2772282
		
