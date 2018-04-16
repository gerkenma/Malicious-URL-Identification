# Malicious-URL-Identification

## Team

**Developer**
> Mark Gerken

**Professor:**
> Kevin Scannell, PhD.

## Requirements
- Python 3.3+ (Tested on Python 3.6.4)

## Set up

1. Install [Python](https://www.python.org/downloads/)
2. If not installed, install Virtualenv with `pip install virtualenv`
3. Navigate in a terminal window to the root directory of the project and set up a virtual environment with `virtualenv venv`
4. Activate the virtual environment with `venv\Scripts\activate` on a Windows machine or `soruce venv\bin\activate`
	- *More complete directions can be found [here](https://virtualenv.pypa.io/en/stable/userguide/#activate-script)*
5. Install the required Python packages with `pip install -r requirements.txt`

## Testing

Dataset retrieved from https://www.sysnet.ucsd.edu/projects/url/. Data is an uncompressed archive of anonymized 120-day dataset consisting of about 2.4 million URLs and 3.2 million features. Each day corresponds to a DayX.svm file (where X is an integer from 0 to 120). URLs are labeled +1 if they are malicious, and -1 if they are benign.
