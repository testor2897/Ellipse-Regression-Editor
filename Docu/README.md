﻿# Fit a 2D ellipse to given data points

This Python program is used for fitting data points to an 2D ellipse.
Several different approaches are used and compared to each other. It is programmed in Python 3.7.9. (windows version).
The parameters of the found regression ellipse can be edited, loaded and saved.

## Files
| File | Description |
|------------------|------------------------------------------|
| *fitEllipse.py* | Main program with the fitting algorithms (Normal start) |
| *start_fitEllipse.py* | Start of program in a venv environment (it will install all requirements in a venv environment) |
| [requirements.txt](../requirements.txt "") |	List of requirements (tkintertable is optional but highly recommended) |
| *mainDialog.py* | Main dialog (using tkinter + tkintertable) |
| *editor.py* | Result and editor dialog (using matplotlib) |
| *editorConfig.py* | Editor configuration file (global variables) |
| *fileDialogs.py* | File dialogs (open, save, ...) |
| *quarticFormula.py* | Formulas for quartic equations (including cardanic and quadratic cases) |
| *Icon_fitEllipse.ico* | Icon file for main dialog |
| *fitEllipse.spec* | Spec file for Pyinstaller Packager (windows version) |
| *Docu* (directory)| Documentation files (start ReadMe.html or Readme.md) |
| *Package* (directory)| Packaged version of program (windows EXE only) |
| *Sample_Data* (directory)| Examples for data points and parameter sets (including data points) |


## Start with *fitEllipse.py*
 
The program is started directly with fitEllipse.py (requirements see above).

If *tkintertable* is installed a GUI is shown:

![GUI](GUI_Example.png)

After the regression is started with button 
> Start Regression
 
the results are displayed via matplotlib:
![Result](Result_Example.png)

If *tkintertable* is not installed a console is used (lesser options):
![Command Line](Command_Line.png)

After the regression results via matplotlib are closed a summary is shown in the console:
![Result - Console](Result_Console.png)


## Start with *Start_fitEllipse.py*
The program is started within a venv enviropnment. The requirements are installed via pip (downloading libraries). This kind of start will take its time...

The advantage of this start type is that no manual installation of Python modules is neccessary. Extra installation is performed automatically in a separate virtual envorinment (venv). Internet Access is needed.


## Start with *fitEllipse.exe* (Package directory)
The program is started as standalone EXE (windows only). No Python is neccessary.
The EXE was packaged with pyinstaller.


## Main dialog
The main dialog:
![GUI](GUI_Details.png)
is divided into 4 areas:

### Selection area
In this area you can:
- select different example point sets (stored for the session),
- load data points from csv- or txt-file (examples below),
- create a new set of data points or
- change number of data points (appending empty points or deleting points).

### Preview area
In this area you will see a preview of the data points if the are validated (see below).

### Data table area
In this are you can edit data points (even scientific format is accepted, commas are translated into points).
In normal cases the input is validated automatically otherways *Validate Data* command can be issued manually.

