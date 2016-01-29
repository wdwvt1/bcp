## Code for behavioral classification and prediction.

# Use
-----
The general use case follows 4 steps:
	1) Data is collected from the Promethion sensors in flat file format output by the Expdata program. This data is parsed with functions in bcp/parse.py and saved to numpy arrays. 
	2) Measurements parsed in step 1 are processed to remove noise using the functions in bcp/{preprocess.py,util.py,stats.py}. For instance, weight measurements are highly variable based on (by discussion with Promethion representatives) temperature fluctuations, animal movement, etc. We might remove these fluctuations and replace these data points with linear interpolation between spanning points which we trust.
	3) Processed data which has had artifacts removed is now passed to functions in bcp/feature_extraction.py which extract/calculate behavioral features. An example of a feature that might be calculated is 'number of feeding events in the last 10 minutes'. 
	4) Features extracted in step 3 are passed to a classifying method which attempts to distinguish the state of the mouse generating/emitting the behavioral feature/character. The code for these calculations is found in bcp/classify.py.

# Conventions
-------------
### `windows`
All functions which calculate a quantity over a `window` use a definition of
window that is inclusive of the point at which the window is centered. For
example:

sequence = abcdefghijk

window = 5
first_average = (a+b+c+f+g)/5

window = 4
first_average = (a+b+c+d)/4

Many functions which operate on a `window` are not defined for every input
point. For example, calculating a moving average is impossible at the edges of
a vector. Functions which would be unable to calculate these edge or boundary
points allow different treatment of the boundary points.

# Ethoscan
----------
The functions in the `bcp/ethoscan.py` module exist to parse and recreate the
calculations that would be made by the Promethion ethoscan.

