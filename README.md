Formatted for TouchDesigner, this script will compare two audio streams and return the time offset between the two.

It may only run in Python 3.7 or 3.8. Librosa has a dependency issue somewhere last I checked.

You will probably need to use TD PIP to make sure you have the correct dependencies installed:

```
def onStart():
	print('STARTING PIP')
	print('PIP setuptools')
	op('td_pip').Import_Module("setuptools", pip_name = "setuptools")
	print('PIP librosa')
	op('td_pip').Import_Module("librosa", pip_name = "librosa")
	print('PIP matplotlib')
	op('td_pip').Import_Module("matplotlib", pip_name = "matplotlib")
	print('COMPLETE')
	return
```
