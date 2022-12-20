pkg:
	python3 setup.py sdist bdist_wheel
clean:
	rm -r build dist glm_lib.egg-info
upload:
	twine upload dist/*
