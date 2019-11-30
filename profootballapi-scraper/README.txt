DESCRIPTION
-----------
This program retrieves historical NFL data from https://profootballapi.com and stores it in the
folder `scraped_data` in your working directory. You can see an example of the output in this repo.

INSTALLATION
------------
- Make sure Python 3.6 or higher is installed
- Sign up for an account at https://profootballapi.com
- Retrieve your API key for your profootballapi account
- Replace `FAKE API KEY` on line 6 in `scraper/__init__.py` with your account's API key
- Install dependencies with 'pip install -r requirements.txt`
- Everything is ready to go!

EXECUTION
---------
- Note: This will take a long time to complete, if you want a short version, change `2020` on
  line 323 in `scraper/__init__.py` to 2010 in order to run the script for a single year of data.
- Run the program in shell with: `python3 scraper/__init__.py`
