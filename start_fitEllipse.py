import sys, os, tempfile, venv
from pathlib import Path

pathSep = os.path.sep


def activate_venv(venvDir=""):
  # -*- coding: utf-8 -*-
  """Activate virtualenv for current interpreter:
  This can be used when you must use an existing Python interpreter, not the virtualenv bin/python.
  """
  import inspect, os, site, sys

  try:
    thisFile = os.path.abspath(__file__)
  except NameError:
    thisFile = os.path.abspath(inspect.stack()[0][1])
  thisDir = os.path.dirname(thisFile)

  # if no venvDir was given use script path
  if venvDir == "":
    baseDir = thisDir
    # strip away the bin part from the thisDir, plus the path separator
    if (baseDir[-4:].lower() == pathSep + "bin"):
      baseDir = baseDir[: -4]  
  else:
    baseDir = venvDir
  
  scriptDir = baseDir + pathSep + "Scripts"
  # Print the script directory of venv
  print("script directory of venv: " + scriptDir)
  
  # prepend scriptDir to PATH
  os.environ["PATH"] = scriptDir + os.pathsep + os.environ.get("PATH", "")
  os.environ["VIRTUAL_ENV"] = baseDir  # virtual env is right above bin or script directory

  sys.real_prefix = sys.prefix
  sys.prefix = baseDir
  
  return
  # add the virtual environments libraries to the host python import mechanism
  ##prev_length = len(sys.path)
  ##for lib in "__LIB_FOLDERS__".split(os.pathsep):
  ##    path = os.path.realpath(os.path.join(thisDir, lib))
  ##    site.addsitedir(path.encode('utf-8').decode("utf-8") if "__DECODE_PATH__" else path)
  ##sys.path[:] = sys.path[prev_length:] + sys.path[0:prev_length]


def get_installed_packages():
  import pkg_resources

  dists = [d for d in pkg_resources.working_set]
  return dists
  

# Print the current working directory
sourceDir = os.getcwd()
print("current working directory: " + sourceDir)

# Print the temp directory
tempDir = tempfile.gettempdir()
os.chdir(tempDir)
print("temp directory: " + tempDir)

# create virtual environment "EllipseFit" in temp directory
venvDir = tempDir + pathSep + "EllipseFit"
venv.create(venvDir, with_pip=True, system_site_packages=False)

# Print the venv directory
print("venv directory: " + venvDir)

# activate virtual environment "EllipseFit" - for child processes only!
activate_venv(venvDir)

# change to sourceDir
os.chdir(sourceDir)

# start child processes within virtual environment 
#import subprocess
#echo_cmd = "pip install -r requirements.txt ".split()
#try:
#  output = subprocess.run(echo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#except:
#  print(output.stderr)

try:
  os.system("pip install -q -r requirements.txt")
except:
  print("PIP-Installation failed:\n", output.stderr)
  
os.system("python " + sourceDir + pathSep + "fitEllipse.py")

##import is_venv_active

##for actName in get_installed_packages():
##  print(actName)
  
##import numpy