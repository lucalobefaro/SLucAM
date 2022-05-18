echo ATE:
python2 evaluate_ate.py ../data/datasets/tum_xyz/groundtruth.txt ../data/datasets/tum_xyz/SLucAM_results.txt --plot tum_xyz_ate_plot.png

echo

echo RPE:
python2 evaluate_rpe.py ../data/datasets/tum_xyz/groundtruth.txt ../data/datasets/tum_xyz/SLucAM_results.txt --fixed_delta --plot tum_xyz_rpe_plot.png
