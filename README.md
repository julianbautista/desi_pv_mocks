# desi_pv_mocks

config.py
- reads yaml configuration file 

bgs_clus.py
- loops over all mocks to create clustering catalogs
- only needs BGS_base mocks and the real BGS clustering catalog

bgs_spec.py
- loops over all mocks to add properties from data
- uses all galaxies from base mock (full sky)
- less than 1 min per mock 
- 12 min for 27 realizations


fp_full.py phase real 
- one mock at a time
- requires bgs_spec.py to have been run
- requires data FP catalog 
- 14 min per mock 

tf_full.py -p phase -r real 
- requires tf clustering catalog from data 
- 20 min per mock 


fp_clus.py 
- requires mock_fp_full_claude.py to have been run
- loops over all mocks 

tf_clus.py
- requires data TF clustering catalog. 
- requires mock_fp_full_claude.py to have been run
- loops over all mocks 

