"""Create and manage file dialogs."""
import os
import re
import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename


class MyFileDialog(tk.Tk):
    """Definiton of file dialog window class."""

    def __enter__(self):
        """Enter the runtime context of the class object."""
        self.withdraw()
        return self


    def __exit__(self, type, value, traceback):
        """Exit the runtime context of the class object."""
        self.destroy()
        self.quit()


    def get_open_path(self, **kwargs):
        """Show open file dialog."""
        return get_open_path(**kwargs)


    def get_save_path(self, **kwargs):
        """Show save file dialog."""
        return get_save_path(**kwargs)


    def get_folder_path(self, **kwargs):
        """Show select directory dialog."""
        return get_folder_path(**kwargs)
        

try:
    def get_open_path(self=None, **kwargs):
        """Show open file dialog."""
        if ("filetypes" not in kwargs):
            kwargs["filetypes"] = (("csv files", "*.csv"), ("all files", "*.*"))
        if ("title" not in kwargs):
            kwargs["title"] = 'Open Sample Data'
        if ("initialdir" not in kwargs):
            kwargs["initialdir"] = os.getcwd()

        path = "" + askopenfilename(**kwargs)
        if path != "":
            os.chdir(os.path.dirname(path))
        return path


    def get_save_path(self=None, **kwargs):
        """Show save file dialog."""
        if ("filetypes" not in kwargs):
            kwargs["filetypes"] = (("csv files", "*.csv"), ("all files", "*.*"))
        if ("title" not in kwargs):
            kwargs["title"] = 'Save Sample Data'
        if ("initialdir" not in kwargs):
            kwargs["initialdir"] = os.getcwd()
        if ("confirmoverwrite" not in kwargs):
            kwargs["confirmoverwrite"] = True

        path = "" + asksaveasfilename(**kwargs)
        if path != "":
            os.chdir(os.path.dirname(path))

        # Make wanted file extension i.e. ".csv" is added
        if "." in path:
            actExtension = "." + path.rsplit(".", 1)[1]
            allExtensions = re.sub(r'\*', '', kwargs["filetypes"][0][1]).split()

            if actExtension not in allExtensions:
                actExtension = ""

        else:
            actExtension = ""

        if path != "" and actExtension == "":
            wantedExtension = re.sub(r'\*', '', kwargs["filetypes"][0][1].split(maxsplit=1)[0])
            path = path + wantedExtension
        return path


    def get_folder_path(self=None, **kwargs):
        """Show select directory dialog."""
        if ("title" not in kwargs):
            kwargs["title"] = 'Select directory'
        if ("initialdir" not in kwargs):
            kwargs["initialdir"] = os.getcwd()

        folder = "" + askdirectory(**kwargs)
        return folder

except Exception:
    print("Problem occured during file dialogs ...\n")


if __name__ == '__main__':
    filetypes = (("csv files", "*.csv"), ("all files", "*.*"))


    # usage if main window is not existing
    # main window is hidden
    fileDialog = MyFileDialog(None)
    with fileDialog as fd:
        print(fd.get_open_path(filetypes=filetypes))
        print(fd.get_save_path())
        print(fd.get_folder_path())


    # usage if main window is already existing
    # otherwise a tiny main window is created
    print(get_open_path(filetypes=filetypes))
    print(get_save_path())
    print(get_folder_path())