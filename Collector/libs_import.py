import re
import ast
from datetime import timedelta
import csv
from apiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date, timedelta
from googleapiclient.errors import HttpError
import matplotlib.pyplot as plt