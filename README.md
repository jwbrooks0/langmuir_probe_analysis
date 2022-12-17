# langmuir_probe_analysis
Single Langmuir Probe Analysis


This code implements a single Langmuir Probe analysis and is primarily based on the work in reference [1].



=== Notes ===
	 * This work is provided "as is" with no gaurantees as to its accuracy or the presence of bugs.  
	 * There are a lot of "nuances" associated with this code and Langmuir probe analysis in general.  I.e. this code can be confusing, make assumptions invalid for your application, there are many "conventions" concerning Langmuir probe analysis, etc.  
	 * PLEASE read through all references and code's docstring first before submitting any questions to me.  This being said, please let me know if you discover any bugs or have other suggestions for improvements.  
	 * This code (at present) does not calculate the final ion density as discussed in [1].  
	 * V_plasma can be calculated in two ways in this code.  First is using the derivative of a smoothed electron current.  Note that this method does not always work particularly if SNR is low, there aren't enough points, and if the sheath is 'thick'.  The second method uses the definition of V_plasma in Lamguir probe theory.  While potentially less accurate, this method is much more robust. 
	 * If you need to debug the code (or debug how the code processes your data), set vebose=True and plot_misc=True.   
		

=== References ===
	 * [1] Lobbia and Beal, "Recommended Practice for Use of Langmuir Probes in Electric Propulsion Testing" https://doi.org/10.2514/1.B35531
	 * [2] Merlino, "Understanding Langmuir probe current-voltage characteristics" https://aapt.scitation.org/doi/10.1119/1.2772282
		