echo ATE:
python2 evaluate_ate.py ../data/datasets/tum_desk/groundtruth.txt ../data/datasets/tum_desk/SLucAM_results.txt --plot tum_desk_ate_plot.png

echo

echo RPE:
python2 evaluate_rpe.py ../data/datasets/tum_desk/groundtruth.txt ../data/datasets/tum_desk/SLucAM_results.txt --fixed_delta --plot tum_desk_rpe_plot.png
