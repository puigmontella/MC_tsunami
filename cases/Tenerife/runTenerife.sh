python ../../Python/1_geomorphology.py

python ../../Python/2_area_maker.py  --plot-overview 
python ../../Python/2_transect_maker.py  --plot-overview 


python ../../Python/3_loading_PGA.py 
python ../../Python/3_loading_PGA_areas.py 

python ../../Python/4_translational_mc.py --keep-fs-checks --n-trials 100000 --n-jobs 100
python ../../Python/4_translational_mc_areas.py --keep-fs-checks --n-trials 100000 --n-jobs 100

python ../../Python/4_rotational_mc.py --keep-fs-checks --n-trials 100000 --n-jobs 100
python ../../Python/4_rotational_mc_areas.py --keep-fs-checks --n-trials 100000 --n-jobs 100


python ../../Python/5_pf_from_rpmin.py --level areas  --types any
python ../../Python/5_pf_from_rpmin.py --level transects  --types any

# python ../../Python/5_pf_from_rpmin.py --level areas  --types rot
# python ../../Python/5_pf_from_rpmin.py --level transects  --types rot


python3 ../../Python/6_coastal_runup.py --source-level areas --types any
python3 ../../Python/6_coastal_runup.py --source-level transects --types any

# python3 ../../Python/6_coastal_runup.py --source-level areas --types rot
# python3 ../../Python/6_coastal_runup.py --source-level transects --types rot

python ../../Python/7_maps_and_figures.py --types trans,rot --sources transects,areas,both --eq-rps 100 --fail-mechs trans,rot
