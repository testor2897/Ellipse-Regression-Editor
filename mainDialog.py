"""Create and manage main GUI window."""
import tkinter as tk
from tkinter import simpledialog, INSERT

try:
    from tkintertable import TableCanvas

    ttLoaded = True
except Exception:
    ttLoaded = False
    print("\ntkintertable not installed!\n")

import csv
import os
from pathlib import Path

import numpy as np

from fileDialogs import get_open_path, get_save_path

standaloneDialog = False

if not ("xSample" in globals()):
    xSample = None
    ySample = None
    titleSample = None
x = None
y = None
title = None
actID = 0
noValidation = False


if not ("is_number" in globals()):
    def is_number(text):
        """
        Check, whether <text> represents a number.

            Parameters:
                text (str): A string

            Returns:
                bool value (True := text can be converted to number)
        """
        try:
            float(text)  # für Float- und Int-Werte
            return True
        except ValueError:
            return False


try:
    class MyTable(TableCanvas):
        """Definiton of modified TableCanvas."""
        def clearData(self, evt=None):
            """Don't Handle clear data events."""
            return
            
            
        def handle_arrow_keys(self, event):
            """Handle arrow keys press in table."""

            # only handle cell events
            if not (isinstance(event.widget, tk.Entry)):
                return
            # Get key information
            actKey = event.keysym
            row = self.currentrow
            col = self.currentcol

            if actKey == 'Right' or actKey == 'Left':
                return

            state = event.state
            # Manual way to get the modifiers
            # ctrl  = (state & 0x4) != 0
            # alt   = (state & 0x8) != 0 or (state & 0x80) != 0
            shift = (state & 0x1) != 0

            x,y = self.getCanvasPos(row, 0)
            if x == None:
                return

            if actKey == 'Up':
                if row == 0:
                    return
                else:
                    self.currentrow  = row - 1
            elif actKey == 'Down':
                if row >= self.rows - 1:
                    return
                else:
                    self.currentrow  = row + 1
            elif actKey == 'Tab' and not (shift):
                if col >= self.cols - 1:
                    if row < self.rows - 1:
                        self.currentcol = 0
                        self.currentrow  = row + 1
                    else:
                        return
                else:
                    self.currentcol  = col + 1
            elif actKey == 'Tab' and shift:
                if col == 0:
                    if row == 0:
                        return
                    else:
                        self.currentcol = self.cols - 1
                        self.currentrow = row - 1
                else:
                    self.currentcol  = col - 1
            self.drawSelectedRect(self.currentrow, self.currentcol)
            coltype = self.model.getColumnType(self.currentcol)
            if coltype == 'text' or coltype == 'number':
                self.delete('entry')
                self.drawCellEntry(self.currentrow, self.currentcol)
            return


    def get_number(self=None, actTitle=None):
        """
        Get user input of number (dialog window).

            Parameters:
                actTitle (str): title for dialog

            Returns:
                number (int): user input
        """
        global standaloneDialog

        if actTitle is None:
            actTitle = self
            self = None
        if actTitle is None:
            actTitle = ""
        root = tk.Tk()
        root.withdraw()
        standaloneDialog = True
        number = simpledialog.askinteger("Input", "Input an Integer", minvalue=5, maxvalue=60)
        standaloneDialog = False
        return number

    def rgb2tk(r, g, b):
        """Convert RGB values to hex RGB string."""
        return f"#{r:02x}{g:02x}{b:02x}"

    class MyApp(tk.Tk):
        """Definiton of main window class."""

        def __init__(self, parent):
            """Initialize class."""
            tk.Tk.__init__(self, parent)
            self.parent = parent
            self.mainWidgets()
            self.protocol("WM_DELETE_WINDOW", self.cancel)
            # make Esc exit the program
            self.bind("<Escape>", self.cancel)
            self.bind("<Return>", self.startRegression)
            self.unbind_all("<<PrevWindow>>")
            self.focus_force()


        def __enter__(self):
            """Enter the runtime context of the class object."""
            return self


        def __exit__(self, type, value, traceback):
            """Exit the runtime context of the class object."""
            return False


        def cancel(self, event=None):
            """Destroy and quit object when cancel event is triggered."""
            global x, y, title

            x = None
            y = None
            title = None
            self.destroy()
            self.quit()


        def mainWidgets(self):
            """Create and configure main window GUI."""
            # don't assume that self.parent is a root window.
            # instead, call `winfo_toplevel to get the root window
            self.winfo_toplevel().title("Elliptical Regression")
            self.winfo_toplevel().iconbitmap("Icon_fitEllipse.ico")
            self.geometry("1000x800")
            self.modus = "build"

            bFont = ("Arial Bold", 12)
            nFont = ("Arial", 10)

            self.hFrame = tk.Frame(master=self)
            self.hFrame.pack(side="top", fill="x", expand=True, padx=20, pady=20)

            self.label1 = tk.Label(self.hFrame, text="Selection:", anchor="w", font=bFont)
            self.label1.pack(side="top", fill="x", expand=True, padx=10)

            self.listbox = tk.Listbox(self.hFrame, activestyle="none", width=40)
            self.listbox.pack(side="left", fill="y", expand=False, padx=10)

            self.canvasFrame = tk.Frame(master=self.hFrame, bg=rgb2tk(255, 255, 204))
            self.canvasFrame.pack(side="top", fill="x", padx=10)
            self.canvas = tk.Canvas(self.canvasFrame)
            self.canvas.pack(side="top", fill="x")

            self.hFrame2 = tk.Frame(master=self, height=90)
            self.hFrame2.pack(side="top", fill="x", expand=True, padx=20, pady=20)

            self.label2 = tk.Label(self.hFrame2, text="Data Table:", anchor="w", font=bFont)
            self.label2.pack(side="top", fill="x", expand=True, padx=10)

            self.gridTitle = tk.StringVar()
            self.textbox = tk.Entry(self.hFrame2, textvariable=self.gridTitle, font=nFont)
            self.textbox.pack(side="top", fill="x", expand=True, padx=12)

            self.gridFrame = tk.Frame(self.hFrame2)
            self.gridFrame.pack(side="left", padx=10)
            self.grid = MyTable(
                parent=self.gridFrame,
                data=None,
                rows=0,
                cols=6,
                height=70,
                showkeynamesinheader=True,
                selectedcolor="white",
                rowselectedcolor="white",
            )
            self.model = self.grid.model
            self.model.addRow(key="x")
            self.model.addRow(key="y")

            self.grid.bgcolor = "white"
            self.grid.rowheight = 30

            table_width = 0
            self.grid.adjustColumnWidths()
            for col in range(self.grid.cols):
                colname = self.model.getColumnName(col)
                if colname in self.model.columnwidths:
                    w = self.model.columnwidths[colname]
                else:
                    w = self.grid.cellwidth
            self.model.columnwidths[colname] = w
            table_width += w
            self.grid.width = table_width
            self.grid.show()

            self.status = tk.Label(
                master=self,
                text="Data points to be validated " + "(press any command button).",
                anchor="w",
            )
            self.status.pack(side="top", fill="x", expand=True, padx=30)
            self.hFrame3 = tk.Frame(master=self)
            self.hFrame3.pack(side="top", fill="x", expand=True, padx=20, pady=20)

            self.valBtn = tk.Button(
                self.hFrame3, text="Validate Data", command=self.validateData, width=15
            )
            self.valBtn.pack(side="left", padx=10)
            self.loadBtn = tk.Button(
                self.hFrame3, text="Load Data", command=self.loadData, width=15
            )
            self.loadBtn.pack(side="left", padx=10)
            self.saveBtn = tk.Button(
                self.hFrame3, text="Save Data", command=self.saveData, width=15
            )
            self.saveBtn.pack(side="left", padx=10)
            self.storeBtn = tk.Button(
                self.hFrame3, text="Store as Sample", command=self.storeSample, width=15
            )
            self.storeBtn.pack(side="left", padx=10)
            self.startBtn = tk.Button(
                self.hFrame3,
                text="Start Regression",
                command=self.startRegression,
                width=15,
            )
            self.startBtn.pack(side="left", padx=10)
            self.cancelBtn = tk.Button(
                self.hFrame3, text="Cancel", command=self.cancel, width=15
            )
            self.cancelBtn.pack(side="left", padx=10)

            # build Listbox
            self.buildListbox()
            if actID < maxID:
                self.changeData(actID)
            else:
                # select first data set
                self.changeData(0)

            self.listbox.bind("<<ListboxSelect>>", self.OnSelect)
            self.grid.bind_all("<KeyRelease>", self.OnEdit)
            self.grid.bind_all("<Return>", self.startRegression)
            self.textbox.bind("<KeyRelease>", self.OnTitlechange)  # keyup
            self.grid.tablecolheader.bind("<Button-3>", self.validateData)
            self.grid.bind("<Button-3>", self.validateData)


        def OnTitlechange(self, event):
            """
            Get user input of title (textbox change).

                Parameters:
                    event (event): <KeyRelease> event on textbox
            """
            global titleSample, title

            title = self.textbox.get()
            if actID < maxID:
                titleSample[actID] = title
                self.buildListbox()
                self.Title = str(actID + 1) + ". Sample (" + title + ")"
            else:
                self.Title = title


        def OnEdit(self, event):
            """
            Get user input for data points (grid change).

                Parameters:
                    event (event): <KeyRelease> event on grid
            """
            global title, actID

            caller = event.widget
            if caller in [self.textbox]:
                return

            if self.modus != "ready":
                self.modus = "ready"
                return

            actTitle = self.Title
            if actID < maxID:
                actID = maxID
                if title[::-1][:11] != " - modified"[::-1]:
                    title = title + " - modified"
                if actTitle[::-1][:11] != " - modified"[::-1]:
                    self.Title = actTitle + " - modified"

            self.gridTitle.set(title)
            self.validateData(None)
            caller.focus_force()


        def OnSelect(self, event):
            """
            Get user input from example selection (<<ListboxSelect>> select).

                Parameters:
                    event (event): <<ListboxSelect>> event on listbox
            """
            actSelection = event.widget.curselection()
            if actSelection:
                index = actSelection[0]
                data = event.widget.get(index)
                self.status["text"] = str(index) + " " + data
                self.changeData(index)
            else:
                self.status["text"] = ""


        def buildListbox(self):
            """Build listbox."""
            global maxID

            if xSample is None:
                maxID = -1
            else:
                maxID = len(xSample)

            self.listbox.delete(0, "end")
            for i in range(maxID):
                self.listbox.insert("end", str(i + 1) + ". Sample (" + titleSample[i] + ")")
            self.listbox.insert("end", "Load Data")
            self.listbox.insert("end", "New Data Points")
            self.listbox.insert("end", "Change Number of Data Points")


        def changeData(self, actSelection):
            """
            Cange data points.

                Parameters:
                    actSelection (int): number of selected example (listbox)
                                        or
                                        actSelection > maxID
                                        (Load Data, New Data Points, Change Number of Data Points)
            """
            global x, y, title, actID

            actLen = None
            doValidation = False
            actID = actSelection
            if actSelection < maxID:
                x = xSample[actSelection]
                y = ySample[actSelection]
                actLen = len(x)

                title = titleSample[actSelection]
                self.gridTitle.set(title)
                self.Title = str(actSelection + 1) + ". Sample (" + title + ")"

                doValidation = True
            elif self.listbox.get(actSelection) == "Load Data":
                wildcard = (
                    ("Text files", "*.txt *.asc *.csv *.prn"),
                    ("all files", "*.*"),
                )
                actTitle = "Regression: Open Sample Data"
                filename = get_open_path(filetypes=wildcard, title=actTitle)

                if filename != "":
                    with open(filename, "r") as csvfile:
                        try:
                            headerstart = csvfile.readline().replace(";",",").replace(" ",",").replace("\t",",").split(",", 1)[0]
                            headerExists = not (is_number(headerstart))
                            if headerExists:
                                if headerstart == "a":
                                    headerLines = 3
                                else:
                                    headerLines = 1
                            else:
                                headerLines = 0
                                csvfile.seek(0)

                            actLine = csvfile.readline()
                            actDelimiter = (
                                csv.Sniffer()
                                .sniff(actLine, delimiters=" \t;")
                                .delimiter
                            )
                        except Exception:
                            csvfile.seek(0)
                            try:
                                actDelimiter = (
                                    csv.Sniffer().sniff(actLine, delimiters=",").delimiter
                                )
                            except Exception:
                                self.status["text"] = "Could not determine delimiter!"
                                self.status.config(fg="red")
                                return
                    try:
                        data = np.genfromtxt(
                            filename,
                            delimiter=actDelimiter,
                            skip_header=headerLines,
                            dtype=str,
                        )
                    except Exception:
                        self.status["text"] = "ERROR reading file: " + filename
                        self.status.config(fg="red")
                        return

                    rows, columns = data.shape
                    if columns > rows:
                        data = data.T
                    x = data[:, 0]
                    y = data[:, 1]
                    actLen = len(x)
                    title = Path(filename).name
                    self.gridTitle.set(title)
                    self.Title = title

                    doValidation = True
            elif self.listbox.get(actSelection) == "New Data Points":
                actLen = get_number(self, "New Data Points")
                if actLen is not None:
                    x = [""] * (actLen)
                    y = [""] * (actLen)
                    actLen = len(x)

                    title = "New Data"
                    self.gridTitle.set(title)
                    self.Title = "New data:"

                    doValidation = False
            elif self.listbox.get(actSelection) == "Change Number of Data Points":
                if "x" in globals():
                    oldLen = len(x)
                else:
                    self.status.status["text"] = "Please create new data points!"
                    self.status.config(fg="red")
                    return

                actLen = get_number(self, "Change Number of Data Points")
                for column in range(oldLen):
                    x[column] = self.model.getValueAt(0, column)
                    y[column] = self.model.getValueAt(1, column)
                if actLen is not None:
                    oldLen = len(x)
                    if actLen < oldLen:
                        x = x[0:actLen]
                        y = y[0:actLen]
                    if actLen > oldLen:
                        x = np.hstack((x, [""] * (actLen - oldLen)))
                        y = np.hstack((y, [""] * (actLen - oldLen)))

                    if title[::-1][:11] != " - modified"[::-1]:
                        title = title + " - modified"

                    self.gridTitle.set(title)
                    self.Title = title

                    doValidation = False
            else:
                return

            if actLen is None:
                return

            gridLen = self.model.getColumnCount()
            if actLen < gridLen:
                # delete columns:
                for column in range(gridLen - actLen):
                    self.model.deleteColumn(gridLen - column - 1)

            if actLen > gridLen:
                # append columns:
                for column in range(actLen - gridLen):
                    self.model.addColumn()

            for column in range(actLen):
                self.model.setValueAt(str(x[column]), 0, column)
                self.model.setValueAt(str(y[column]), 1, column)
                self.model.columnlabels[column] = str(column + 1)

            table_width = 0
            self.grid.redrawTable()
            for col in range(self.grid.cols):
                colname = self.model.getColumnName(col)
                if colname in self.model.columnwidths:
                    w = self.model.columnwidths[colname]
                else:
                    w = self.grid.cellwidth
                self.model.columnwidths[colname] = w
                table_width += w
            self.grid.config(width=table_width)
            self.grid.width = table_width
            # self.update_idletasks()
            self.grid.redrawTable()

            if doValidation:
                self.validateData(None)
            else:
                self.status["text"] = ("Data points to be validated "
                                       "(press 'Validate Data or 'Start Regression').")
                self.status.config(fg="black")
                self.canvas.delete("all")

            self.grab_set()
            self.grid.focus_force()
            self.grid.drawCellEntry(0, 0)


        def validateData(self, event=None):
            """
            Validate point data.

                Parameters:
                    event (event): not used

                Returns:
                    isValid (bool): True := data points are valid
            """
            global x, y, noValidation

            if noValidation:
                noValidation = False
                return False

            isValid = True
            errList = ""
            actLen = self.model.getColumnCount()
            tempX = [None] * actLen
            tempY = [None] * actLen

            for column in range(actLen):
                tempX[column] = self.model.getValueAt(0, column).replace(",", ".")
                tempY[column] = self.model.getValueAt(1, column).replace(",", ".")
                try:
                    self.model.setValueAt(tempX[column], 0, column)
                    tempX[column] = float(tempX[column])
                except Exception:
                    isValid = False
                    errList += " x[" + str(column + 1) + "]"
                try:
                    self.model.setValueAt(tempY[column], 1, column)
                    tempY[column] = float(tempY[column])
                except Exception:
                    isValid = False
                    errList += " y[" + str(column + 1) + "]"

            self.grid.redrawTable()
            if not isValid:
                errList = " Please correct cells:" + errList

            if actLen < 5:
                errList += " Not enough data points! (<5)"
                isValid = False

            if isValid:
                self.status["text"] = "Data points are valid."
                self.status.config(fg="green")
                x = np.array(tempX)
                y = np.array(tempY)
                self.update()

                cWidth = self.canvas.winfo_width()
                cHeight = self.canvas.winfo_height()
                self.canvas.delete("all")
                self.canvas.configure(bg=rgb2tk(255, 255, 204))
                # x axis
                self.canvas.create_line(
                    35, cHeight - 40, cWidth - 40, cHeight - 40, arrow=tk.LAST
                )
                # y axis
                self.canvas.create_line(40, cHeight - 35, 40, 40, arrow=tk.LAST)
                # title
                self.canvas.create_text(cWidth / 2, 10, text=self.Title)

                formatCode = "{:2g}"
                sFont = ("Arial", 6)
                # x marker
                xMin = np.min(x)
                xMax = np.max(x)
                self.canvas.create_text(cWidth / 2, cHeight - 20, text="x")
                self.canvas.create_text(
                    40, cHeight - 20, text=formatCode.format(xMin), font=sFont
                )
                self.canvas.create_text(
                    cWidth - 40, cHeight - 20, text=formatCode.format(xMax), font=sFont
                )
                # y marker
                yMin = np.min(y)
                yMax = np.max(y)
                self.canvas.create_text(20, cHeight / 2, text="y")
                self.canvas.create_text(
                    20, cHeight - 40, text=formatCode.format(yMin), font=sFont
                )
                self.canvas.create_text(
                    20, 40, text=formatCode.format(yMax), font=sFont
                )
                # data points
                mX = (cWidth - 80) / (xMax - xMin)
                mY = (cHeight - 80) / (yMax - yMin)
                for i in range(actLen):
                    deltaX = x[i] - xMin
                    actX = 40 + deltaX * mX
                    deltaY = y[i] - yMin
                    actY = cHeight - 40 - deltaY * mY
                    self.canvas.create_oval(
                        actX, actY, actX, actY, width=6, outline="blue"
                    )
            else:
                if len(errList) > 80:
                    lastPos = errList.find("]", 80) + 1
                    errList = errList[:lastPos] + " ..."
                self.status["text"] = "Data points are invalid:" + errList
                self.status.config(fg="red")
                self.canvas.delete("all")
            self.startBtn.focus_set()
            return isValid


        def loadData(self):
            """Redirect loadData button to changeData routine."""
            self.changeData(maxID)


        def saveData(self, event=None):
            """Save point data to file."""
            global title

            isValid = self.validateData(None)
            if isValid:
                wildcard = (("csv files", "*.csv"), ("all files", "*.*"))
                actTitle = "Regression: Save Sample Data"

                filename = get_save_path(filetypes=wildcard, title=actTitle)
                if filename != "":
                    try:
                        np.savetxt(
                            filename, np.vstack((x, y)).T, delimiter=",", fmt="%g"
                        )
                    except Exception:
                        self.status["text"] = "ERROR writing file: " + filename
                        self.status.config(fg="red")
                        return
            # if successful saved, refresh preview
            if actID < maxID:
                title = Path(filename).name
                self.gridTitle.set(title)

            self.Title = title
            self.validateData(None)


        def storeSample(self, event=None, isNewSample=True):
            """
            Keep current point data as sample (not permanent).

                Parameters:
                    isNewSample (bool): sample is not existing yet
            """
            global title

            isValid = self.validateData(None)
            if isValid:
                if isNewSample or (title != titleSample[maxID - 1]):
                    xSample.append(x)
                    ySample.append(y)
                    titleSample.append(title)
                else:
                    xSample[maxID - 1] = x
                    ySample[maxID - 1] = y
                    titleSample[maxID - 1] = title
                self.buildListbox()
                self.changeData(maxID - 1)
            else:
                self.status["text"] = "ERROR storing sample: data not valid!"
                self.status.config(fg="red")
                self.canvas.delete("all")


        def startRegression(self, event=None):
            """Start Regression and close main window."""
            global noValidation

            # only continue if not called by <return> key event from standalone dialog
            if standaloneDialog:
                return

            isValid = self.validateData(None)
            if isValid:
                if actID >= maxID:
                    self.storeSample(None, isNewSample=False)
                noValidation = True
                self.quit()
                self.destroy()


# if no tkintertable is installed
except Exception:
    print("Proceeding without tkintertable...\n")


def selectPointData(xDataList=None, yDataList=None, titleDataList=None):
    """Start main GUI window or console mode."""
    global xSample, ySample, titleSample, x, y, title

    xSample = xDataList
    ySample = yDataList
    titleSample = titleDataList

    if xSample is None:
        actCount = -1
    else:
        actCount = len(xSample)

    # if tkintertable is loaded show GUI
    if ttLoaded:
        application = MyApp(None)
        with application as app:
            app.mainloop()
    else:
        actText = "L (load data file) or Q (Quit): "
        if actCount > 0:
            actText = (
                "Please enter sample number (1 ... " + str(actCount) + ") or " + actText
            )
        actInput = input(actText).upper().strip()

        try:
            if actInput == "Q":
                # exit without exception
                os._exit(0)
            elif actInput == "L":
                filename = input("Please enter filename incl. path: ").strip()
                with open(filename, "r") as csvfile:
                    print("loading...")
                    try:
                        actDelimiter = (
                            csv.Sniffer()
                            .sniff(csvfile.read(), delimiters=" \t;")
                            .delimiter
                        )
                    except Exception:
                        csvfile.seek(0)
                        try:
                            actDelimiter = csv.Sniffer().sniff(csvfile.read()).delimiter
                        except Exception:
                            print("Could not determine delimiter  => Abort")
                            # exit without exception
                            os._exit(0)
                    try:
                        data = np.genfromtxt(
                            filename, dtype=str, delimiter=actDelimiter
                        )
                        x = np.char.replace(data[:, 0], ",", ".").astype(np.float)
                        y = np.char.replace(data[:, 1], ",", ".").astype(np.float)
                        title = filename
                    except Exception:
                        print("ERROR reading file: " + filename + "  => Abort")
                        # exit without exception
                        os._exit(0)
            else:
                actSelection = int(actInput)
                if actSelection < 1 or actSelection > actCount:
                    print("Selection out of range: number should be between 1 and ",
                           str(actCount), " => Abort")
                    # exit without exception
                    os._exit(0)

                # get selected sample data
                x = xSample[actSelection - 1]
                y = ySample[actSelection - 1]
                title = titleSample[actSelection - 1]
        except Exception:
            print("Invalid input => Abort")
            quit()
    return [x, y, title]
