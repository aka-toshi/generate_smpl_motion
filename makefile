main: demo.py
#	python -m demo --img_path ${0}.png --json_path data/random_keypoints.json
	python -m demo --img_path ${0}.png
	python smplmake.py ${0}

