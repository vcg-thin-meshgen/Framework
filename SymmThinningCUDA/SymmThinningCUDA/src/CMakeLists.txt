
add_library(skeletonize STATIC attachment.cu  
	neighbor.cu  
	thinning.cu  
	clique.cu  
	thinning_details.cu)
target_compile_features(skeletonize PUBLIC cxx_std_11)
set_property(TARGET skeletonize PROPERTY CUDA_SEPARABLE_COMPILATION ON)



