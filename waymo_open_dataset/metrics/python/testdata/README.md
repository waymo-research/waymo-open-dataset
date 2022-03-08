## Test data to verify OKS metric

Json file `oks_testdata.json` contains a list of dictionaries in the following
 format:

key | value
--- | ---
ground_truth | a dictionary with keypoints
prediction | a dictionary with keypoints
oks | scalar

dictionaries with keypoints in turn have the following format:

key | value
--- | ---
keypoints | A list with N*3 float values representing 2D coordinates and visibility values, e.g.: `x_0, y_0, v_0, ..., x_n, y_n, v_n`
bbox | a list with 4 float values (`left`, `top`, `width`, `height`)
area | a float number equal to `width * height` of the bbox

2D coordinates were randomly generated, expected OKS values were computed using `computeOks` function from [cocoeval.py](https://github.com/matteorr/coco-analyze/blob/9eb8a0a9e57ad1e592661efc2b8964864c0e6f28/pycocotools/cocoeval.py#L203).

