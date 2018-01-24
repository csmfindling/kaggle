To obtain hdf5, download code of authors on website:
http://www.ee.cuhk.edu.hk/~xgwang/projectpage_structured_feature_pose.html

then enter file pose-v2 and launch Data_prepare.m in Matlab and then run "ConvertLMDB.sh" to generate LMDB data

then go in external/data/lsp : here is the data. Let us note path_lsp the path leading to the lsp data, for me it is "../../pretraining_data/lsp/" (I moved them)

next step, go to the kaggle-drivers/pretraining/ folder and open build_hdf5.py. Change lines 15 to 18 adequetaly with lsp_path and line 44. finally launch script python and wait for the result
